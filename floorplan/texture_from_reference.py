"""
floorplan/texture_from_reference.py — Reference-texture stage.

Extracts wall textures directly from the reference photo by projecting each
visible wall's corners into image space and warping the corresponding region
into a flat rectangular texture (same approach as Gen3DSR / photo-texture
baking).

Saves (never overwrites existing files):
  <scene>/wall_back_texture_ref.png
  <scene>/wall_left_texture_ref.png
  <scene>/wall_right_texture_ref.png  (if visible)
  <scene>/walls_metadata_ref.json     (copy of walls_metadata.json with updated paths)
  <scene>/render_ref_texture.png      (full re-render with reference textures)

Usage (standalone):
    python -m floorplan.texture_from_reference --scene outputs/Demo/living_room9

Usage (pipeline — add "texture_from_reference" to --stages):
    python main.py --rerun-dir outputs/Demo/living_room9 --stages texture_from_reference
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from floorplan.texture_cleanup import (
    clean as clean_texture_defects,
    detect_openings,
    shear_angle,
    apply_vshear,
)


# ── Camera helpers (matches render_room.py's architectural convention) ─────────

def _build_camera(cam: dict, img_w_px: int, img_h_px: int):
    """Return (pos, right, fwd_h, tilt_tan, fx, cx, cy) ready for projection."""
    pos     = np.array(cam["position_m"], dtype=float)
    look_at = np.array(cam["look_at_m"],  dtype=float)

    fwd = look_at - pos
    fwd /= max(np.linalg.norm(fwd), 1e-9)

    # Horizontal right — world-up locked to avoid Manhattan-roll artifacts
    right_raw = np.cross(fwd, np.array([0., 1., 0.]))
    rn = np.linalg.norm(right_raw)
    right = right_raw / rn if rn > 1e-9 else np.array([1., 0., 0.])

    # Horizontal forward (strips y tilt) — keeps verticals vertical
    fwd_h_raw = np.array([fwd[0], 0., fwd[2]])
    fwd_h_n   = np.linalg.norm(fwd_h_raw)
    fwd_h     = fwd_h_raw / fwd_h_n if fwd_h_n > 1e-9 else fwd.copy()
    tilt_tan  = float(fwd[1]) / max(fwd_h_n, 1e-9)

    W_px  = int(cam.get("width_px",  img_w_px))
    H_px  = int(cam.get("height_px", img_h_px))
    hfov  = float(cam["hfov_deg"])
    fx    = W_px / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W_px / 2.0, H_px / 2.0

    # If reference image resolution differs from camera params, rescale
    if img_w_px != W_px or img_h_px != H_px:
        sx, sy = img_w_px / W_px, img_h_px / H_px
        fx *= sx
        cx *= sx
        cy *= sy

    return pos, right, fwd_h, tilt_tan, fx, cx, cy


def _project(pt_world, pos, right, fwd_h, tilt_tan, fx, cx, cy):
    """Project a 3-D world point to pixel (x, y) — same formula as render_room.

    Returns None when the point is behind the camera (zc ≤ 0).
    """
    d   = np.asarray(pt_world, dtype=float) - pos
    xc  = float(np.dot(d, right))
    zc  = float(np.dot(d, fwd_h))
    yc  = float(d[1]) - tilt_tan * zc
    if zc < 0.01:
        return None
    return np.array([cx + fx * xc / zc,
                     cy - fx * yc / zc], dtype=np.float32)


# ── Room geometry ─────────────────────────────────────────────────────────────

def _room_dims(out_dir: Path, cam: dict) -> tuple[float, float, float]:
    """Return (W_m, D_m, H_m).

    Primary source is the calibrated box geometry (walls.obj) — the SAME mesh
    that render_room rasterizes, so the backprojected wall textures are sized to
    exactly what gets rendered, and any depth-refined extent change (see
    depth_wall_refine) flows through to the backprojection automatically.
    Falls back to walls_metadata.json / wall_context / camera-Z when the obj is
    missing or degenerate.
    """
    obj_path = out_dir / "walls.obj"
    if obj_path.exists():
        try:
            v = np.array([[float(t) for t in l.split()[1:4]]
                          for l in open(obj_path) if l.startswith("v ")])
            lo, hi = v.min(0), v.max(0)
            W_m, D_m, H_m = float(hi[0] - lo[0]), float(hi[2] - lo[2]), float(hi[1] - lo[1])
            if W_m > 0.5 and D_m > 0.5 and H_m > 0.5:
                return W_m, D_m, H_m
        except Exception:
            pass

    W_m = D_m = H_m = None

    meta_path = out_dir / "walls_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        W_m = meta.get("back",  {}).get("length_m") or None
        H_m = meta.get("back",  {}).get("height_m") or None
        D_m = (meta.get("left",  {}).get("length_m") or
               meta.get("right", {}).get("length_m") or None)
        if D_m and D_m < 0.1:
            D_m = None  # 0 means "not measured"

    wc = cam.get("wall_context", {})
    if not W_m:
        W_m = float(wc.get("back", {}).get("length_m") or 4.0)
    if not H_m:
        H_m = float(wc.get("back", {}).get("height_m") or 2.7)
    if not D_m:
        # Fallback: camera Z position ≈ room depth (camera sits near front wall)
        D_m = float(cam["position_m"][2]) * 1.05

    return float(W_m), float(D_m), float(H_m)


# Corners ordered [bottom-left, bottom-right, top-right, top-left]
# as seen from inside the room looking at the wall.
def _wall_corners(wall: str, W: float, D: float, H: float) -> list[np.ndarray]:
    """Return wall corners ordered so _warp_wall produces a texture whose UV
    convention matches render_room._face_uv:
      corners[3] → texture top-left  → UV(0, 0) = floor, low-u edge
      corners[0] → texture bot-left  → UV(0, 1) = ceiling, low-u edge
    i.e. floor row at top of texture (ty=0) and ceiling at bottom (ty=H-1),
    matching _face_uv where UV.y = world_y/tile (floor y=0 → ty=0).
    """
    if wall == "back":   # Z-facing, UV.x = world_x, UV.y = world_y
        return [np.array([0, H, 0]), np.array([W, H, 0]),
                np.array([W, 0, 0]), np.array([0, 0, 0])]
    if wall == "left":   # X-facing, UV.x = world_z, UV.y = world_y
        return [np.array([0, H, 0]), np.array([0, H, D]),
                np.array([0, 0, D]), np.array([0, 0, 0])]
    if wall == "right":  # X-facing, UV.x = world_z, UV.y = world_y
        return [np.array([W, H, 0]), np.array([W, H, D]),
                np.array([W, 0, D]), np.array([W, 0, 0])]
    if wall == "front":  # Z-facing, UV.x = world_x, UV.y = world_y
        return [np.array([0, H, D]), np.array([W, H, D]),
                np.array([W, 0, D]), np.array([0, 0, D])]
    raise ValueError(f"Unknown wall: {wall!r}")


def _facing_camera(wall: str, pos: np.ndarray, W: float, D: float) -> bool:
    x, _, z = pos
    return {"back": z > 0, "front": z < D,
            "left": x > 0, "right": x < W}.get(wall, False)


# ── Core warp ─────────────────────────────────────────────────────────────────

_PX_PER_M = 256   # output texture resolution


def _stretch_mask(M: np.ndarray, W_tex: int, H_tex: int,
                  rel_thresh: float = 0.4) -> np.ndarray:
    """Locate over-stretched grazing pixels (the deep wall corner, where the
    homography maps a thin source sliver onto a large texture area and smears
    the panels). Per-pixel source-area sampled per texture pixel = |Jacobian| of
    the inverse homography; pixels below rel_thresh × median are over-stretched.

    Returned as a SEPARATE mask (uint8) — NOT folded into the out-of-bounds
    invalid mask, so it never changes the global hole-fill method. The caller
    fills it with a gentle local TELEA instead."""
    Minv = np.linalg.inv(M)
    xs, ys = np.meshgrid(np.arange(W_tex), np.arange(H_tex))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)[None]
    src = cv2.perspectiveTransform(pts, Minv).reshape(H_tex, W_tex, 2)
    dux = np.gradient(src[..., 0], axis=1); dvx = np.gradient(src[..., 1], axis=1)
    duy = np.gradient(src[..., 0], axis=0); dvy = np.gradient(src[..., 1], axis=0)
    area = np.abs(dux * dvy - duy * dvx)
    pos = area[area > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    if med <= 0:
        return np.zeros((H_tex, W_tex), dtype=np.uint8)
    mask = (area < rel_thresh * med).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def _warp_from_src_pts(img_bgr: np.ndarray, src_pts: np.ndarray,
                        wall_w_m: float, wall_h_m: float,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspective-warp 4 source points → flat texture + out-of-bounds mask.

    src_pts: (4,2) float32  [ceil-left, ceil-right, floor-right, floor-left].
    Returns (warped_raw, invalid_mask, stretch_mask). invalid_mask is uint8,
    1=missing pixel (caller completes via _complete_uv_texture); stretch_mask is
    uint8, 255=over-stretched grazing pixel (caller fills with gentle local TELEA).
    """
    W_tex = max(64, int(wall_w_m * _PX_PER_M))
    H_tex = max(64, int(wall_h_m * _PX_PER_M))

    dst_pts = np.array([[0,         H_tex - 1],
                        [W_tex - 1, H_tex - 1],
                        [W_tex - 1, 0        ],
                        [0,         0        ]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_bgr, M, (W_tex, H_tex),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REPLICATE)

    ih_src, iw_src = img_bgr.shape[:2]
    mask_src = np.zeros((ih_src, iw_src), dtype=np.uint8)
    clipped = src_pts.copy()
    clipped[:, 0] = clipped[:, 0].clip(0, iw_src - 1)
    clipped[:, 1] = clipped[:, 1].clip(0, ih_src - 1)
    cv2.fillConvexPoly(mask_src, clipped.astype(np.int32), 255)
    warped_mask = cv2.warpPerspective(mask_src, M, (W_tex, H_tex),
                                      flags=cv2.INTER_NEAREST)
    invalid = (warped_mask < 128).astype(np.uint8)
    stretch = _stretch_mask(M, W_tex, H_tex)
    # don't fill the same pixel twice — out-of-bounds wins.
    stretch[invalid > 0] = 0
    return warped, invalid, stretch


def _tile_fill(tex: np.ndarray, invalid: np.ndarray,
               min_band_px: int = 24) -> np.ndarray | None:
    """Synthesize a large missing region by mirror-tiling the real texture band
    instead of smooth-inpainting it. Walls are vertical panels, so the visible
    part is a contiguous column band; repeating it (mirrored, to hide seams)
    across the missing columns gives crisp panels rather than a blurry fill.

    Returns the filled texture, or None if there isn't enough valid texture to
    tile (caller then falls back to inpainting)."""
    H, W = invalid.shape
    valid = (invalid == 0)
    col_valid = valid.mean(axis=0)
    good = col_valid > 0.5

    # widest contiguous run of valid columns → the source band to tile.
    best0 = best1 = 0
    cur = None
    for x in range(W + 1):
        if x < W and good[x]:
            cur = x if cur is None else cur
        elif cur is not None:
            if x - cur > best1 - best0:
                best0, best1 = cur, x
            cur = None
    c0, c1 = best0, best1
    bw = c1 - c0
    if bw < min_band_px:
        return None

    band = tex[:, c0:c1].copy()
    band_inv = invalid[:, c0:c1]
    if band_inv.any():                       # heal small holes inside the band
        band = cv2.inpaint(band, band_inv, 9, cv2.INPAINT_TELEA)

    blocks, w, flip = [], 0, False
    while w < W:
        blocks.append(band[:, ::-1] if flip else band)
        w += bw
        flip = not flip
    tiled = np.concatenate(blocks, axis=1)[:, :W]

    out = tex.copy()
    m = invalid > 0
    out[m] = tiled[m]
    return out


def _largest_valid_rect(valid: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-valid axis-aligned rectangle in a bool mask → (y0, x0, y1, x1).
    Standard max-rectangle-in-histogram sweep, O(H·W)."""
    H, W = valid.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)                      # area, y0, x0, y1, x1
    for y in range(H):
        heights = np.where(valid[y], heights + 1, 0)
        stack = []                              # (start_x, height)
        for x in range(W + 1):
            h = int(heights[x]) if x < W else 0
            start = x
            while stack and stack[-1][1] > h:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, y - sh + 1, sx, y + 1, x)
                start = sx
            stack.append((start, h))
    _, y0, x0, y1, x1 = best
    return y0, x0, y1, x1


def _floor_exemplar_tile(tex: np.ndarray, invalid: np.ndarray) -> np.ndarray | None:
    """Fill the floor's missing region by mirror-tiling a CLEAN rectangular patch
    taken from the straightened (backprojected) visible floor. The visible floor
    is an irregular region, so we extract the largest fully-valid rectangle as the
    exemplar and tile it over the whole map — uniform planks, no NS smear."""
    valid = (invalid == 0)
    y0, x0, y1, x1 = _largest_valid_rect(valid)
    ph, pw = y1 - y0, x1 - x0
    if ph < 16 or pw < 16:
        return None
    patch = tex[y0:y1, x0:x1]
    H, W = invalid.shape

    rows, yy, fy = [], 0, False
    while yy < H:
        block = patch[::-1] if fy else patch
        cols, xx, fx = [], 0, False
        while xx < W:
            cols.append(block[:, ::-1] if fx else block)
            xx += pw
            fx = not fx
        rows.append(np.concatenate(cols, axis=1)[:, :W])
        yy += ph
        fy = not fy
    tiled = np.concatenate(rows, axis=0)[:H]

    out = tex.copy()
    m = invalid > 0
    out[m] = tiled[m]
    return out


def _floor_uniform_tile(tex: np.ndarray, invalid: np.ndarray,
                        contam_thresh: float = 30.0) -> np.ndarray | None:
    """Make a clean UNIFORM floor: the floor is one material (marble/tile), and
    the backprojected floor is mostly missing + contaminated (wall-bleed,
    reflections) + warp-distorted, which tiling only the holes leaves as a
    shattered patchwork. Instead, extract the largest CLEAN patch — valid pixels
    whose Lab colour is close to the dominant floor colour (so wallpaper/trim/
    reflection bleed is excluded) — and mirror-tile it across the WHOLE texture.
    Returns None if there isn't a clean patch to tile from."""
    valid = invalid == 0
    if valid.sum() < 200:
        return None
    lab = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB).astype(np.float32)
    dom = np.median(lab[valid].reshape(-1, 3), axis=0)
    dist = np.sqrt(((lab - dom) ** 2).sum(axis=2))
    clean = (valid & (dist < contam_thresh)).astype(np.uint8)
    # erode slightly so the patch stays away from contamination/edges
    clean = cv2.erode(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    y0, x0, y1, x1 = _largest_valid_rect(clean.astype(bool))
    ph, pw = y1 - y0, x1 - x0
    if ph < 16 or pw < 16:
        return None
    patch = tex[y0:y1, x0:x1]
    H, W = invalid.shape
    rows, yy, fy = [], 0, False
    while yy < H:
        block = patch[::-1] if fy else patch
        cols, xx, fx = [], 0, False
        while xx < W:
            cols.append(block[:, ::-1] if fx else block)
            xx += pw
            fx = not fx
        rows.append(np.concatenate(cols, axis=1)[:, :W])
        yy += ph
        fy = not fy
    return np.concatenate(rows, axis=0)[:H]


def _complete_uv_texture(tex: np.ndarray, invalid: np.ndarray,
                          wall: str, prefer_tile: bool = False) -> np.ndarray:
    """Fill missing regions in a warped UV texture so the surface is complete.

    invalid: uint8 mask, 1 = pixel came from outside the source image and needs
             inpainting.  The function is always called even when invalid has no
             set pixels (returns tex unchanged), so callers don't need to check.

    Strategy:
      - Dilate mask by 1px to cover hard BORDER_REPLICATE edge seams.
      - Small gap (< 25%): TELEA radius-9 — fast, good for thin borders.
      - Large gap (≥ 25%) on a wall: mirror-tile the real panel band across the
        gap (crisp panels, not blur); fall back to median-seed + NS-40 if there
        isn't enough valid texture to tile. Floor always uses NS-40.
    """
    if not invalid.any():
        return tex

    missing_frac = float(invalid.sum()) / invalid.size
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    invalid_d = cv2.dilate(invalid, k, iterations=1)

    big = missing_frac >= 0.25
    method = "TELEA-9"
    if wall == "floor" and big:
        # Two floor-fill strategies, chosen per scene (auto-detecting which is
        # needed proved unreliable — tile-grout, marble veining, contamination and
        # shadows all look alike to colour/edge/chroma metrics):
        #   default (uniform): _floor_uniform_tile mirror-tiles ONE clean low-variance
        #     patch over the WHOLE floor. Best for uniform marble whose backprojection
        #     is contaminated/warp-shattered (rgb_003431) — replaces the mess. But it
        #     strips real tile patterns (grout/diamonds) into featureless stripes.
        #   SCENEWEAVE_FLOOR_PRESERVE=1: _floor_exemplar_tile keeps the real visible
        #     pixels and tiles only the holes — preserves a genuine tile/plank pattern
        #     (rgb_003477). Use when the backprojected floor is clean + patterned.
        import os as _os2
        preserve = bool(_os2.environ.get("SCENEWEAVE_FLOOR_PRESERVE"))
        if preserve:
            tiled = _floor_exemplar_tile(tex, invalid_d)   # keep real tiles, fill holes
            if tiled is None:
                tiled = _floor_uniform_tile(tex, invalid_d)
            _m = "tile-floor-preserve"
        else:
            tiled = _floor_uniform_tile(tex, invalid_d)    # uniform/clean replace
            if tiled is None:
                tiled = _floor_exemplar_tile(tex, invalid_d)
            _m = "tile-floor"
        print(f"[ref_tex] floor: mode={'PRESERVE-pattern' if preserve else 'uniform'}")
        if tiled is not None:
            tex = tiled
            method = _m
        else:
            valid_px = tex[invalid == 0].reshape(-1, 3)
            if valid_px.size:
                seed = np.median(valid_px, axis=0).astype(np.uint8)
                tex = tex.copy()
                tex[invalid_d > 0] = seed
            tex = cv2.inpaint(tex, invalid_d, 40, cv2.INPAINT_NS)
            method = "NS-40"
    elif wall != "floor" and (big or prefer_tile):
        tiled = _tile_fill(tex, invalid_d)        # crisp panels across the gap
        if tiled is not None:
            tex = tiled
            method = "tile"
        elif big:
            valid_px = tex[invalid == 0].reshape(-1, 3)
            if valid_px.size:
                seed = np.median(valid_px, axis=0).astype(np.uint8)
                tex = tex.copy()
                tex[invalid_d > 0] = seed
            tex = cv2.inpaint(tex, invalid_d, 40, cv2.INPAINT_NS)
            method = "NS-40"
        else:
            tex = cv2.inpaint(tex, invalid_d, 9, cv2.INPAINT_TELEA)
    elif big:
        valid_px = tex[invalid == 0].reshape(-1, 3)
        if valid_px.size:
            seed = np.median(valid_px, axis=0).astype(np.uint8)
            tex = tex.copy()
            tex[invalid_d > 0] = seed
        tex = cv2.inpaint(tex, invalid_d, 40, cv2.INPAINT_NS)
        method = "NS-40"
    else:
        tex = cv2.inpaint(tex, invalid_d, 9, cv2.INPAINT_TELEA)

    print(f"[ref_tex] {wall}: uv-complete  missing={missing_frac*100:.1f}%  "
          f"method={method}")
    return tex


def _warp_wall(img_bgr: np.ndarray, corners_world: list[np.ndarray],
               cam_params: tuple, wall_w_m: float, wall_h_m: float,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Perspective-warp the wall's image quad → (warped, invalid, stretch).

    Returns None if any corner is behind the camera.
    """
    pos, right, fwd_h, tilt_tan, fx, cx, cy = cam_params
    src_pts = []
    for c in corners_world:
        px = _project(c, pos, right, fwd_h, tilt_tan, fx, cx, cy)
        if px is None:
            return None
        src_pts.append(px)
    return _warp_from_src_pts(img_bgr, np.array(src_pts, dtype=np.float32),
                               wall_w_m, wall_h_m)


# ── Manhattan-detection-guided wall segmentation ──────────────────────────────

def _back_wall_y_at_x(image_x: float, world_y: float,
                       cam_params: tuple) -> float | None:
    """Image y for a back-wall point (z=0) at the given image column and world height.

    Solves for the 3-D world x that projects to image_x on the back wall (z=0),
    then projects world_y to get the image y.  This lets us compute correct
    floor/ceiling image y at a detected corner x without needing the 3-D mesh x.
    """
    pos, right, fwd_h, tilt_tan, fx, cx, cy = cam_params
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])

    # Solve for x3d (with z=0):
    #   image_x = cx + fx * (right[0]*(x3d-px) + right[2]*(0-pz))
    #                      / (fwd_h[0]*(x3d-px) + fwd_h[2]*(0-pz))
    # Let u = x3d - px, r = (image_x - cx) / fx:
    #   u*(r*fwd_h[0] - right[0]) = right[2]*(-pz) - r*fwd_h[2]*(-pz)*(-1) ...
    # Simplified (right[1]=fwd_h[1]=0 for horizontal vectors):
    #   x_cam = right[0]*u - right[2]*pz
    #   z_cam = fwd_h[0]*u - fwd_h[2]*pz
    #   r*(fwd_h[0]*u - fwd_h[2]*pz) = right[0]*u - right[2]*pz
    #   u*(r*fwd_h[0] - right[0]) = r*fwd_h[2]*pz - right[2]*pz ... wait:
    #   r*fwd_h[0]*u - r*fwd_h[2]*pz = right[0]*u - right[2]*pz
    #   u*(r*fwd_h[0] - right[0]) = r*fwd_h[2]*pz - right[2]*pz
    #   Rearranged: u = pz*(r*fwd_h[2] - right[2]) / (r*fwd_h[0] - right[0])
    r = (image_x - cx) / fx
    denom = r * float(fwd_h[0]) - float(right[0])
    if abs(denom) < 1e-9:
        return None
    u = pz * (r * float(fwd_h[2]) - float(right[2])) / denom
    x3d = u + px

    d = np.array([x3d - px, world_y - py, 0.0 - pz])
    z_cam = float(np.dot(d, fwd_h))
    if z_cam < 0.01:
        return None
    y_cam = float(d[1]) - tilt_tan * z_cam
    return float(cy - fx * y_cam / z_cam)


def _snap_to_nearest_corner(image_x: float,
                             detected_xs: list[float]) -> float:
    """Return the detected corner x closest to image_x."""
    if not detected_xs:
        return image_x
    return min(detected_xs, key=lambda dx: abs(dx - image_x))


def _build_detected_wall_quads(
    scene_dir: Path,
    cam_params: tuple,
    W_m: float, D_m: float, H_m: float,
    img_w: int, img_h: int,
) -> dict[str, np.ndarray] | None:
    """Build per-wall image-space source quads using a hybrid strategy.

    Strategy
    --------
    * Project the 3-D mesh corners to 2-D as usual — this gives correct *y*
      values (floor/ceiling lines) because camera tilt/height are well estimated.
    * For each projected corner, snap its *x* value to the nearest detected
      vertical wall line from corners.json — this corrects horizontal
      misalignment caused by mesh imprecision without disturbing y.

    Falls back to raw 3-D projection for any wall whose projected x has no
    close detected counterpart (snap tolerance = 10% of image width).

    Returns dict wall_name -> (4,2) float32 [ceil-left, ceil-right,
    floor-right, floor-left], or None if corners.json is missing / unusable.
    """
    corners_path = scene_dir / "corners.json"
    if not corners_path.exists():
        return None

    cdata = json.loads(corners_path.read_text())
    raw_corners = cdata.get("corners", [])
    if not raw_corners:
        return None

    detected_xs = [float(c["x"]) for c in raw_corners]
    snap_tol    = img_w * 0.10   # 10% of image width

    pos, right, fwd_h, tilt_tan, fx, cx, cy = cam_params

    def _proj_pt(world_pt) -> np.ndarray | None:
        return _project(world_pt, pos, right, fwd_h, tilt_tan, fx, cx, cy)

    def _snap_x(px: float) -> float:
        snapped = _snap_to_nearest_corner(px, detected_xs)
        return snapped if abs(snapped - px) <= snap_tol else px

    cam_x, cam_z = float(pos[0]), float(pos[2])
    fh0, fh2 = float(fwd_h[0]), float(fwd_h[2])

    def _z_visible_max(x_wall: float, margin: float = 0.08) -> float:
        """Max z at which (x_wall, *, z) is still in front of the camera.

        Derived from fwd_h · (x_wall-cam_x, 0, z-cam_z) = 0 → z = cam_z - fh0*(x_wall-cam_x)/fh2.
        """
        if abs(fh2) < 1e-9:
            return cam_z - margin
        return float(cam_z - fh0 * (x_wall - cam_x) / fh2) - margin

    quads: dict[str, np.ndarray] = {}
    # Also track the visible depth for each side wall so run() can size the texture.
    quad_depths: dict[str, float] = {}

    # ── Back wall  (z=0, x: 0→W_m) ───────────────────────────────────────────
    # Project each corner independently.  For corners behind the camera (extreme
    # yaw), fall back to image edge.  The projected x is snapped to the nearest
    # detected concave corner so the boundary aligns with the Manhattan junction.
    # Y values always come from _back_wall_y_at_x which is robust to extreme yaw.
    proj_cl = _proj_pt(np.array([0,   H_m, 0]))
    proj_fl = _proj_pt(np.array([0,   0,   0]))
    proj_cr = _proj_pt(np.array([W_m, H_m, 0]))
    proj_fr = _proj_pt(np.array([W_m, 0,   0]))

    if proj_cl is not None:
        back_left_x = _snap_x(proj_cl[0])
    elif proj_fl is not None:
        back_left_x = _snap_x(proj_fl[0])
    else:
        back_left_x = 0.0

    if proj_cr is not None:
        back_right_x = _snap_x(proj_cr[0])
    elif proj_fr is not None:
        back_right_x = _snap_x(proj_fr[0])
    else:
        back_right_x = float(img_w - 1)

    y_cl = _back_wall_y_at_x(back_left_x,  H_m, cam_params)
    y_fl = _back_wall_y_at_x(back_left_x,  0.0, cam_params)
    y_cr = _back_wall_y_at_x(back_right_x, H_m, cam_params)
    y_fr = _back_wall_y_at_x(back_right_x, 0.0, cam_params)

    if back_right_x - back_left_x > 4 and None not in (y_cl, y_fl, y_cr, y_fr):
        quads["back"] = np.array([
            [back_left_x,  y_cl],
            [back_right_x, y_cr],
            [back_right_x, y_fr],
            [back_left_x,  y_fl],
        ], dtype=np.float32)
        print(f"[ref_tex] back: anchor x={back_left_x:.0f} (snapped from "
              f"{proj_cl[0]:.0f}) → quad x=[{back_left_x:.0f}, {back_right_x:.0f}]"
              if proj_cl is not None else
              f"[ref_tex] back: quad x=[{back_left_x:.0f}, {back_right_x:.0f}]")
    else:
        print(f"[ref_tex] back: skip (width={back_right_x - back_left_x:.0f}px)")

    # ── Side walls — project real 3-D corners; clip x to image edge ──────────
    # Inner edge = back junction (z=0): always project these directly (they ARE
    # the back-wall corners so they share the same visiblity we just confirmed).
    # Outer edge = front of wall (z=D_m): may land off-screen on x; clip x to
    # the image edge but keep the projected y so the perspective gradient is
    # correct — missing source pixels are filled by the improved inpainting.
    back_left_x  = float(quads["back"][0, 0]) if "back" in quads else back_left_x
    back_right_x = float(quads["back"][1, 0]) if "back" in quads else back_right_x

    # Left wall (x=0)
    p_lcl_in  = _proj_pt(np.array([0, H_m, 0  ]))   # ceil,  back junction
    p_lfl_in  = _proj_pt(np.array([0, 0,   0  ]))   # floor, back junction
    p_lcl_out = _proj_pt(np.array([0, H_m, D_m]))   # ceil,  front (may be off-screen)
    p_lfl_out = _proj_pt(np.array([0, 0,   D_m]))   # floor, front
    if p_lcl_in is not None and p_lfl_in is not None:
        inner_x  = _snap_x(p_lcl_in[0])
        outer_x  = 0.0                               # clip to image left edge
        y_cl_out = float(p_lcl_out[1]) if p_lcl_out is not None else 0.0
        y_fl_out = float(p_lfl_out[1]) if p_lfl_out is not None else float(img_h - 1)
        if inner_x - outer_x > 4:
            quads["left"] = np.array([
                [inner_x, float(p_lcl_in[1])],   # ceil-inner  → texture UV.x=0
                [outer_x, y_cl_out           ],   # ceil-outer  → texture UV.x=1
                [outer_x, y_fl_out           ],   # floor-outer
                [inner_x, float(p_lfl_in[1])],   # floor-inner
            ], dtype=np.float32)
            quad_depths["left"] = inner_x / max(img_w, 1) * D_m
            print(f"[ref_tex] left: quad x=[{outer_x:.0f}, {inner_x:.0f}]  "
                  f"outer_ceil_y={y_cl_out:.0f}")

    # Right wall (x=W_m)
    p_rcl_in  = _proj_pt(np.array([W_m, H_m, 0  ]))
    p_rfl_in  = _proj_pt(np.array([W_m, 0,   0  ]))
    p_rcl_out = _proj_pt(np.array([W_m, H_m, D_m]))
    p_rfl_out = _proj_pt(np.array([W_m, 0,   D_m]))
    if p_rcl_in is not None and p_rfl_in is not None:
        inner_x  = _snap_x(p_rcl_in[0])
        outer_x  = float(img_w - 1)                 # clip to image right edge
        y_cr_out = float(p_rcl_out[1]) if p_rcl_out is not None else 0.0
        y_fr_out = float(p_rfl_out[1]) if p_rfl_out is not None else float(img_h - 1)
        if outer_x - inner_x > 4:
            quads["right"] = np.array([
                [inner_x, float(p_rcl_in[1])],   # ceil-inner  → texture UV.x=0
                [outer_x, y_cr_out           ],   # ceil-outer  → texture UV.x=1
                [outer_x, y_fr_out           ],
                [inner_x, float(p_rfl_in[1])],
            ], dtype=np.float32)
            quad_depths["right"] = (outer_x - inner_x) / max(img_w, 1) * D_m
            print(f"[ref_tex] right: quad x=[{inner_x:.0f}, {outer_x:.0f}]  "
                  f"outer_ceil_y={y_cr_out:.0f}")

    return quads, quad_depths


# ── VLM texture refinement ────────────────────────────────────────────────────

_SURFACE_LABELS = {
    "back":    "back wall",
    "left":    "left side wall",
    "right":   "right side wall",
    "front":   "front wall (behind camera)",
    "floor":   "floor",
    "ceiling": "ceiling",
}


def _refine_texture_with_vlm(tex_bgr: np.ndarray, wall: str,
                             backend: str | None = None) -> np.ndarray | None:
    """Post-process a perspective-warped texture with the configured image-edit
    model.  Fixes blurry regions, neighboring-wall bleed-in, and edge artifacts.

    Args:
        backend: "qwen", "gemini", or None to use SCENEWEAVE_IMG_EDIT env var
                 (defaults to "qwen" if unset).

    Returns a refined BGR ndarray (same resolution) on success, None on failure.
    """
    try:
        import base64
        import io
        from PIL import Image as _PIL
        from Qwen.image_edit_adapter import edit_image
    except ImportError as e:
        print(f"  [ref_tex] VLM refine import error: {e}")
        return None

    rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = tex_bgr.shape[:2]

    # Encode as PNG base64
    pil_img = _PIL.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    surface_desc = _SURFACE_LABELS.get(wall, wall)
    prompt = (
        f"Convert this warped {surface_desc} photo crop into a pure flat texture map "
        f"of ONLY the surface material itself. "
        f"The output must contain ONLY the repeating surface material — "
        f"paint, plaster, wood, tile, fabric, etc. — with zero room context. "
        f"Completely remove: wall edges, corners, adjacent walls, ceiling lines, "
        f"floor lines, furniture, objects, fixtures, shadows, depth cues, and any "
        f"color bleed from neighboring surfaces. "
        f"Fill every pixel uniformly with the dominant surface material and color "
        f"visible in the center of the image. "
        f"Output: flat orthographic texture, uniform neutral lighting, "
        f"no perspective, no depth, no edges, no borders, tileable pattern only."
    )
    neg_prompt = (
        "wall edge, corner, adjacent wall, ceiling, floor line, furniture, objects, "
        "room, interior, shadows, depth, perspective, borders, stripes from other "
        "surfaces, color bleed, architectural elements, dark edges"
    )

    try:
        resp = edit_image({
            "prompt":              prompt,
            "negative_prompt":     neg_prompt,
            "reference_image":     b64,
            "num_inference_steps": 30,
            "true_cfg_scale":      3.5,
        }, backend=backend, timeout=240)
    except Exception as e:
        print(f"  [ref_tex] VLM edit request failed for {wall}: {e}")
        return None

    imgs = resp.get("images", [])
    if not imgs:
        print(f"  [ref_tex] VLM returned no images for {wall}")
        return None

    raw = base64.b64decode(imgs[0])
    refined_pil = _PIL.open(io.BytesIO(raw)).convert("RGB")
    refined_pil = refined_pil.resize((orig_w, orig_h), _PIL.LANCZOS)
    return cv2.cvtColor(np.array(refined_pil), cv2.COLOR_RGB2BGR)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(output_dir: str | Path, bg_image_path: str | Path | None = None,
        refine: bool = False, backend: str | None = None,
        clean: bool = True) -> dict:
    """Extract reference-photo wall textures and produce a re-render.

    Does NOT touch walls_metadata.json or any existing texture files.

    Args:
        output_dir    : Scene directory (must have camera_vggt.json / camera.json
                        and walls.obj).
        bg_image_path : Optional path to a background-only / empty-room image to
                        use as the texture source instead of the scene's own
                        reference photo.  Useful when the reference has furniture
                        occluding the walls.
        clean         : Deterministic defect-removal + delighting on each baked
                        texture (removes cabinet/floor/neighbour bleed and
                        flattens the lighting gradient for uniform tone). On by
                        default; runs before the optional VLM refine.
        refine        : Post-process each texture with a VLM image-edit model.
        backend       : "qwen" | "gemini" | None.  None falls back to the
                        SCENEWEAVE_IMG_EDIT env var, then "qwen".

    Returns:
        dict with keys: textures (wall→path str), metadata_path, render_path.
    """
    out = Path(output_dir)

    # ── camera ────────────────────────────────────────────────────────────────
    cam_path = out / "camera_vggt.json"
    if not cam_path.exists():
        cam_path = out / "camera.json"
    if not cam_path.exists():
        raise FileNotFoundError(f"No camera JSON in {out}")
    cam = json.loads(cam_path.read_text())

    # ── source image (bg_image_path overrides scene's own reference) ──────────
    if bg_image_path is not None:
        ref_img = cv2.imread(str(bg_image_path))
        if ref_img is None:
            raise FileNotFoundError(f"Could not read bg_image: {bg_image_path}")
        print(f"[ref_tex] bg image: {Path(bg_image_path).name}  "
              f"{ref_img.shape[1]}×{ref_img.shape[0]}px")
    else:
        ref_img = None
        for candidate in sorted(out.glob("*.jpg")) + sorted(out.glob("*.jpeg")) + \
                         sorted(out.glob("*.png")) + sorted(out.glob("*.webp")):
            if candidate.stat().st_size > 10_000:
                ref_img = cv2.imread(str(candidate))
                if ref_img is not None:
                    print(f"[ref_tex] reference image: {candidate.name}  "
                          f"{ref_img.shape[1]}×{ref_img.shape[0]}px")
                    break
        if ref_img is None:
            raise FileNotFoundError(f"No reference image found in {out}")
    ih, iw = ref_img.shape[:2]

    # ── camera params ─────────────────────────────────────────────────────────
    cam_params = _build_camera(cam, iw, ih)
    pos = cam_params[0]

    # ── room dims ─────────────────────────────────────────────────────────────
    W_m, D_m, H_m = _room_dims(out, cam)
    print(f"[ref_tex] room  W={W_m:.2f}m  D={D_m:.2f}m  H={H_m:.2f}m")

    # ── existing metadata ─────────────────────────────────────────────────────
    meta_path = out / "walls_metadata.json"
    base_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # ── per-wall extraction ───────────────────────────────────────────────────
    wall_dim = {
        "back":  (W_m, H_m),
        "left":  (D_m, H_m),
        "right": (D_m, H_m),
        "front": (W_m, H_m),
    }
    saved_textures: dict[str, str] = {}

    # Try Manhattan-detection-guided quads first; fall back to 3-D projection.
    # _build_detected_wall_quads also returns visible depths for side walls
    # (clipped to just before the camera) so we can size textures correctly.
    # Default: define each wall's image region from the KNOWN 3D geometry
    # (walls.obj projected through the calibrated camera). This is robust for
    # oblique side walls, where image-based corner detection dips into the floor
    # and shears the rectification. Opt back into detection via env var.
    import os as _os
    if _os.environ.get("SCENEWEAVE_USE_DETECTED_QUADS"):
        _dq_result = _build_detected_wall_quads(out, cam_params, W_m, D_m, H_m, iw, ih)
    else:
        _dq_result = None
    detected_quads, quad_depths = _dq_result if _dq_result is not None else (None, {})

    for wall in ("back", "left", "right", "front"):
        if not _facing_camera(wall, pos, W_m, D_m):
            print(f"[ref_tex] {wall}: not facing camera — skip")
            continue

        ww, wh = wall_dim[wall]
        if detected_quads is not None and wall in detected_quads:
            if wall in quad_depths:
                ww = quad_depths[wall]
            raw, mask, stretch = _warp_from_src_pts(ref_img, detected_quads[wall], ww, wh)
            src = "detected-corners"
        else:
            corners = _wall_corners(wall, W_m, D_m, H_m)
            result = _warp_wall(ref_img, corners, cam_params, ww, wh)
            if result is None:
                print(f"[ref_tex] {wall}: corner behind camera — skip")
                continue
            raw, mask, stretch = result
            src = "3d-projection"

        print(f"[ref_tex] {wall}: warp source={src}")

        if _os.environ.get("WALL_DEBUG") == wall:
            _v = raw.copy(); _v[mask > 0] = (255, 0, 255)
            cv2.imwrite(str(out / f"_wall_{wall}_backproj.png"), _v)
            _s = raw.copy(); _s[stretch > 0] = (0, 255, 0)
            cv2.imwrite(str(out / f"_wall_{wall}_stretch.png"), _s)
            print(f"[ref_tex] {wall} DEBUG: valid={100*(mask==0).mean():.0f}% "
                  f"stretch={100*(stretch>0).mean():.0f}%")

        # Treat large very-dark openings (a doorway / opening to an adjacent dark
        # space baked into the wall plane) as missing, so they're completed with
        # the wall pattern. Lightness-keyed, so coloured accent strips are kept.
        # Dark *decorative* wall panels (e.g. dark-blue upholstered wainscoting)
        # trip the dark-opening detector and get tiled over, splitting clean panels
        # with false seams. SCENEWEAVE_NO_OPENING_FILL=1 skips this for such scenes.
        if _os.environ.get("SCENEWEAVE_NO_OPENING_FILL"):
            opening = np.zeros_like(mask)
        else:
            opening = detect_openings(raw, valid=(mask == 0), which="both")
        has_opening = bool(opening.any())
        if has_opening:
            mask = ((mask > 0) | (opening > 0)).astype(np.uint8)
            print(f"[ref_tex] {wall}: opening-fill {opening.mean()*100:.1f}% "
                  f"(non-wall dark region → wall pattern)")

        tex = _complete_uv_texture(raw, mask, wall, prefer_tile=has_opening)

        # Rectify an oblique wall (off-frame top corner → vertical shear). The
        # shear is only reliably measurable on a CLEANED texture (the raw/just-
        # completed warp's panel edges are too faint for the detector). So probe
        # with a clean pass to measure the angle, then apply the inverse shear to
        # the RAW warp + masks and RE-complete — inpainting into an already-
        # straight frame rather than deskewing after the fill (per design intent).
        # Shear detection needs contrast-enhanced edges, so probe with the
        # full (delight) pass; but it is ONLY used to measure the angle.
        probe = clean_texture_defects(tex, stretch_mask=stretch, light=False) \
            if clean else tex
        theta = shear_angle(probe)
        if theta:
            raw     = apply_vshear(raw,     theta, border_value=0)
            mask    = apply_vshear(mask,    theta, border_value=1, nearest=True)
            stretch = apply_vshear(stretch, theta, border_value=0, nearest=True)
            tex = _complete_uv_texture(raw, mask, wall, prefer_tile=has_opening)
            print(f"[ref_tex] {wall}: deskew {theta:+.1f}° (rectify + re-inpaint)")

        if clean:
            # LIGHT finish: preserve the backprojected pattern, only fill the
            # over-stretched grazing corner (no delight, no defect removal).
            tex = clean_texture_defects(tex, stretch_mask=stretch, light=True)
            print(f"[ref_tex] {wall}: light-finish (real data preserved)")

        if refine:
            refined = _refine_texture_with_vlm(tex, wall, backend=backend)
            if refined is not None:
                tex = refined
                print(f"[ref_tex] {wall}: VLM refinement applied")
            else:
                print(f"[ref_tex] {wall}: VLM refinement failed — using raw warp")

        tex_path = out / f"wall_{wall}_texture_ref.png"
        cv2.imwrite(str(tex_path), tex)
        saved_textures[wall] = str(tex_path)
        print(f"[ref_tex] {wall}: saved {tex.shape[1]}×{tex.shape[0]}px → {tex_path.name}")

    if not saved_textures:
        raise RuntimeError("[ref_tex] No wall textures could be extracted.")

    # ── Inherit textures for walls backprojection had to skip ──────────────────
    # A wall skipped ("not facing camera" / "corner behind camera", e.g. a grazing
    # side wall) otherwise keeps its flat floorplan placeholder — grey for scenes
    # whose floorplan ran before the sampled-placeholder change. Walls in a room
    # almost always share the same material, so copy a successfully-backprojected
    # sibling wall's texture onto each skipped real wall for a consistent result
    # instead of a bare grey panel.
    import shutil as _shutil
    _donor = next((w for w in ("back", "left", "right", "front")
                   if w in saved_textures), None)
    if _donor is not None:
        for _w in ("back", "left", "right", "front"):
            if _w in saved_textures or _w not in wall_dim:
                continue
            try:
                _shutil.copyfile(out / f"wall_{_donor}_texture_ref.png",
                                 out / f"wall_{_w}_texture_ref.png")
                saved_textures[_w] = str(out / f"wall_{_w}_texture_ref.png")
                print(f"[ref_tex] {_w}: backprojection skipped → inherited "
                      f"'{_donor}' wall texture (same room material)")
            except Exception as _e:
                print(f"[ref_tex] {_w}: inherit failed ({_e})")

    # ── floor texture extraction ──────────────────────────────────────────────
    # Use the visible floor region (back wall to just in front of camera).
    # Corners ordered [front-left, front-right, back-right, back-left] so that
    # _warp_wall maps: front-left→(0,H-1), front-right→(W-1,H-1),
    # back-right→(W-1,0), back-left→(0,0)  —  near=bottom / far=top of texture.
    pz = float(pos[2])
    px = float(pos[0])
    fwd_h = cam_params[2]  # horizontal forward unit vector
    fh0, fh2 = float(fwd_h[0]), float(fwd_h[2])
    z_front = max(0.5, pz - 0.3)

    # Clip z_front so that left-edge corners stay in front of the camera.
    if abs(fh2) > 1e-6:
        z_max = pz + (fh0 * (0.0 - px)) / (-fh2) - 0.1
        z_front = min(z_front, z_max)
    z_front = max(0.3, z_front)

    # For yawed cameras, the right-side floor corners can be behind camera even
    # at z_front=0.  Compute x_floor_max: the max world x where z_cam > 0 for
    # both z=0 and z=z_front (the more restrictive of the two near/far edges).
    # z_cam=0 at z=zp for x on the floor: x = px + fh2*(pz-zp)/fh0
    x_floor_max = W_m
    if abs(fh0) > 1e-9 and fh0 < 0:   # camera looks left → right side clips
        x_vis_back  = px + fh2 * pz         / fh0 - 0.05   # at z=0
        x_vis_front = px + fh2 * (pz - z_front) / fh0 - 0.05  # at z=z_front
        x_floor_max = max(0.5, min(W_m, x_vis_back, x_vis_front))

    floor_corners = [
        np.array([0,           0, z_front]),
        np.array([x_floor_max, 0, z_front]),
        np.array([x_floor_max, 0, 0      ]),
        np.array([0,           0, 0      ]),
    ]
    floor_visible_d = z_front  # depth of visible region (back wall → z_front)
    floor_result = _warp_wall(ref_img, floor_corners, cam_params, x_floor_max, floor_visible_d)
    if floor_result is not None:
        floor_raw, floor_mask, _floor_stretch = floor_result  # floor is resized; skip stretch-fix
        # Remove dark wall-bleed sampled at the floor-wall junction (brown stripes
        # from the wood wall). Floor edges aren't trim, so don't protect them.
        _fbleed = detect_openings(floor_raw, valid=(floor_mask == 0),
                                  which="dark", protect_trim=False)
        if _fbleed.any():
            floor_mask = ((floor_mask > 0) | (_fbleed > 0)).astype(np.uint8)
            print(f"[ref_tex] floor: wall-bleed removed {_fbleed.mean()*100:.1f}%")
        if _os.environ.get("FLOOR_DEBUG"):
            _vis = floor_raw.copy()
            _vis[floor_mask > 0] = (255, 0, 255)   # magenta = missing/inpainted region
            cv2.imwrite(str(out / "_floor_backproj_only.png"), _vis)
            print(f"[ref_tex] floor DEBUG: backproject valid={100*(floor_mask==0).mean():.0f}%"
                  f" → _floor_backproj_only.png")
        floor_tex = _complete_uv_texture(floor_raw, floor_mask, "floor")
        # Scale height to full room depth so tile_size_m tiles correctly
        full_H = max(64, int(D_m * _PX_PER_M))
        floor_tex_full = cv2.resize(floor_tex, (floor_tex.shape[1], full_H),
                                    interpolation=cv2.INTER_LANCZOS4)
        if clean:
            floor_tex_full = clean_texture_defects(floor_tex_full, light=True)
            print("[ref_tex] floor: light-finish (pattern preserved)")
        if refine:
            refined_floor = _refine_texture_with_vlm(floor_tex_full, "floor",
                                                      backend=backend)
            if refined_floor is not None:
                floor_tex_full = refined_floor
                print("[ref_tex] floor: VLM refinement applied")
            else:
                print("[ref_tex] floor: VLM refinement failed — using raw warp")
        floor_tex_path = out / "floor_texture_ref.png"
        cv2.imwrite(str(floor_tex_path), floor_tex_full)
        saved_textures["floor"] = str(floor_tex_path)
        print(f"[ref_tex] floor: saved {floor_tex_full.shape[1]}×{full_H}px → {floor_tex_path.name}")
    else:
        print("[ref_tex] floor: corner behind camera — skip")

    # ── ceiling texture extraction ────────────────────────────────────────────
    # The ceiling plane (y=H_m) seen from below.  For an eye-level camera it's
    # grazing / mostly above-frame, so usually only a strip near the back wall is
    # visible.  Ceilings are near-uniform, so we sample the dominant ceiling
    # colour (from the backprojected strip, else the top of the reference photo)
    # and fill the whole ceiling with it — a faithful tint, as one seamless piece
    # instead of the flat-white placeholder.
    ceil_corners = [
        np.array([0,           H_m, z_front]),
        np.array([x_floor_max, H_m, z_front]),
        np.array([x_floor_max, H_m, 0      ]),
        np.array([0,           H_m, 0      ]),
    ]
    ceil_col = None
    ceil_backprojected = False
    ceil_tex_path = out / "ceiling_texture_ref.png"
    _full_Hc = max(64, int(D_m * _PX_PER_M))
    _full_Wc = max(64, int(W_m * _PX_PER_M))
    # Non-uniform ceilings (coffered/recessed/painted panels — e.g. gold insets)
    # lose their pattern under the default median fill.  With
    # SCENEWEAVE_CEILING_BACKPROJECT=1 we keep the perspective-backprojected
    # ceiling (holes completed) exactly like the floor, instead of a flat colour.
    _ceil_bp = _os.environ.get("SCENEWEAVE_CEILING_BACKPROJECT", "") in (
        "1", "true", "yes", "on")
    ceil_result = _warp_wall(ref_img, ceil_corners, cam_params, x_floor_max, floor_visible_d)
    if ceil_result is not None:
        ceil_raw, ceil_mask, _ = ceil_result
        # drop dark wall-bleed sampled at the ceiling-wall junction
        _cbleed = detect_openings(ceil_raw, valid=(ceil_mask == 0),
                                  which="dark", protect_trim=False)
        _cvalid = (ceil_mask == 0) & (_cbleed == 0)
        if _cvalid.mean() > 0.02:
            ceil_col = np.median(ceil_raw[_cvalid].reshape(-1, 3), axis=0)
            print(f"[ref_tex] ceiling: backprojected strip visible="
                  f"{_cvalid.mean()*100:.0f}%")
        if _ceil_bp and _cvalid.mean() > 0.04:
            _cfill = ((ceil_mask > 0) | (_cbleed > 0)).astype(np.uint8)
            if _os.environ.get("CEILING_DEBUG"):
                _cv = ceil_raw.copy(); _cv[_cfill > 0] = (255, 0, 255)
                cv2.imwrite(str(out / "_ceiling_backproj_only.png"), _cv)
                print(f"[ref_tex] ceiling DEBUG: backproject valid="
                      f"{100*(_cfill==0).mean():.0f}% → _ceiling_backproj_only.png")
            # Keep the DIRECT backprojection so captured detail (e.g. coffered gold
            # panels) stays ALIGNED where the warp put it, and INPAINT the UNSEEN
            # region so it blends coherently with the surrounding ceiling. Tiling
            # (_complete_uv_texture) shifts/repeats a one-off architectural ceiling
            # and seams it; a flat solid fill is patchy — inpaint extends the real
            # surround instead.
            _cm = (_cfill > 0).astype(np.uint8)
            if _cm.any() and (_cfill == 0).any():
                ceil_tex = cv2.inpaint(ceil_raw, _cm, 6, cv2.INPAINT_TELEA)
            else:
                ceil_tex = ceil_raw.copy()
            ceil_tex_full = cv2.resize(ceil_tex, (ceil_tex.shape[1], _full_Hc),
                                       interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(ceil_tex_path), ceil_tex_full)
            saved_textures["ceiling"] = str(ceil_tex_path)
            ceil_backprojected = True
            print(f"[ref_tex] ceiling: BACKPROJECTED {ceil_tex_full.shape[1]}×{_full_Hc}px "
                  f"(valid {_cvalid.mean()*100:.0f}%) → {ceil_tex_path.name}")
    if not ceil_backprojected:
        if ceil_col is None:
            # fallback: dominant colour of the top strip of the reference photo
            _top = ref_img[:max(1, int(0.08 * ref_img.shape[0]))]
            ceil_col = np.median(_top.reshape(-1, 3), axis=0)
            print("[ref_tex] ceiling: no backprojected strip — using ref photo top-strip colour")
        ceil_tex_full = np.full((_full_Hc, _full_Wc, 3),
                                np.asarray(ceil_col, dtype=np.uint8), dtype=np.uint8)
        cv2.imwrite(str(ceil_tex_path), ceil_tex_full)
        saved_textures["ceiling"] = str(ceil_tex_path)
        print(f"[ref_tex] ceiling: uniform fill colour(BGR)={np.asarray(ceil_col).astype(int).tolist()} "
              f"→ {ceil_tex_path.name}")

    # ── build walls_metadata_ref.json ─────────────────────────────────────────
    meta_ref = copy.deepcopy(base_meta)
    for wall, tex_path in saved_textures.items():
        if wall == "floor":
            entry = meta_ref.setdefault("floor", {})
            entry["texture_path"] = tex_path
            entry["tile_size_m"]  = x_floor_max   # matches visible extraction width
            # Align the floor planks to the reference: measured against the
            # original/empty-room plank angle, 180° is the orientation that
            # matches (the ref-warp + render axis conventions).
            entry["uv_rotation_deg"] = 180
            continue
        if wall == "ceiling":
            entry = meta_ref.setdefault("ceiling", {})
            entry["texture_path"] = tex_path
            entry["tile_size_m"]  = W_m       # uniform fill → tile size is cosmetic
            entry["uv_rotation_deg"] = 0
            continue
        ww, wh = wall_dim[wall]
        entry = meta_ref.setdefault(wall, {})
        entry["texture_path"] = tex_path
        entry["tile_size_m"]  = ww          # texture covers full wall WIDTH → no h-tiling
        entry["tile_size_v_m"] = wh         # texture covers full wall HEIGHT → V maps 0..1
        # _wall_corners is ordered so floor → ty=0 (texture top), ceiling → ty=TH-1,
        # matching _face_uv (UV.y = world_y → ty=0 at floor).  No rotation needed.
        entry["uv_rotation_deg"] = 0

    meta_ref_path = out / "walls_metadata_ref.json"
    meta_ref_path.write_text(json.dumps(meta_ref, indent=2))
    print(f"[ref_tex] metadata → {meta_ref_path.name}")

    # ── re-render with reference textures using camera_vggt.json ─────────────
    # render_room reads walls_metadata.json AND global fallback textures
    # (wall_texture.png, floor_texture.png, ceiling_texture.png) from texture_dir.
    # We build a temp dir with the ref metadata + copies of the scene's fallbacks
    # so floor/ceiling get their original textures while walls get the ref ones.
    render_path: Path | None = None
    try:
        from floorplan.wall_line.render_room import render_room

        # Always prefer camera_vggt.json for the render
        render_cam = out / "camera_vggt.json"
        if not render_cam.exists():
            render_cam = cam_path

        walls_obj = out / "walls_with_windows.obj"
        if not walls_obj.exists():
            walls_obj = out / "walls.obj"
        if not walls_obj.exists():
            raise FileNotFoundError(f"No walls OBJ found in {out}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)

            # Ref metadata — render_room reads this as walls_metadata.json
            (tmp_p / "walls_metadata.json").write_text(
                json.dumps(meta_ref, indent=2))

            # Copy scene's global fallback textures so floor/ceiling render correctly
            for fname in ("wall_texture.png", "floor_texture.png",
                          "ceiling_texture.png", "wall_front_texture.png"):
                src = out / fname
                if src.exists():
                    shutil.copy2(src, tmp_p / fname)

            # Override with reference textures where available
            for ref_name, fallback_name in [
                ("floor_texture_ref.png",   "floor_texture.png"),
                ("ceiling_texture_ref.png", "ceiling_texture.png"),
            ]:
                ref_src = out / ref_name
                if ref_src.exists():
                    shutil.copy2(ref_src, tmp_p / fallback_name)
                    print(f"[ref_tex] render: using {ref_name} as {fallback_name}")

            render_path = out / "render_ref_texture.png"
            render_room(
                mesh_path=str(walls_obj),
                camera_json_path=str(render_cam),
                out_path=str(render_path),
                texture_dir=str(tmp_p),
                no_legend=True,
                # These textures are real-photo-derived (delit) albedo, so render
                # near full brightness instead of darkening walls to the default
                # ambient (=0.75) — walls have horizontal normals and get no
                # diffuse from the up-facing key/fill lights.
                ambient=0.97,
            )

        print(f"[ref_tex] render → {render_path.name}  "
              f"(camera: {render_cam.name})")
    except Exception as e:
        import traceback
        print(f"[ref_tex] render FAILED: {e}")
        traceback.print_exc()
        render_path = None

    return {
        "textures":      saved_textures,
        "metadata_path": str(meta_ref_path),
        "render_path":   str(render_path) if render_path else None,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scene", required=True,
                    help="Scene output directory (e.g. outputs/Demo/living_room9)")
    ap.add_argument("--bg-image", default=None,
                    help="Optional empty-room / background-only image to use as "
                         "the texture source instead of the scene's reference photo "
                         "(e.g. outputs/empty_rooms_gpt2/living_room9.png)")
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Disable the default deterministic defect-removal + "
                         "delighting on each baked texture.")
    ap.add_argument("--refine", action="store_true",
                    help="Post-process each extracted texture with the image-edit "
                         "model (cleans up edge artifacts and bleed-in from "
                         "neighboring walls). Requires Qwen or Gemini backend.")
    ap.add_argument("--backend", choices=["qwen", "gemini"], default=None,
                    help="Image-edit backend for --refine: 'qwen' (local server, "
                         "default) or 'gemini' (requires GEMINI_API_KEY). "
                         "Overrides the SCENEWEAVE_IMG_EDIT env var.")
    args = ap.parse_args()
    result = run(args.scene, bg_image_path=args.bg_image,
                 refine=args.refine, backend=args.backend, clean=args.clean)
    print("\n[ref_tex] Done.")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
