"""
generate_window_plane.py — create a flat plane GLB for each window/door segment,
                            textured with the (rectified) inpainted image.

Replaces the 3D generator for windows.  the 3D generator reconstructs a thick 3D mesh which adds
unwanted depth and distorts the frame placement.  A thin flat plane is better:
  • Zero depth — lies perfectly flush against the wall.
  • UV-mapped — the full rectified window image is projected onto the face.
  • Aspect ratio preserved — plane dimensions match the inpainted image ratio.

The plane is created in the 3D generator convention:
  • Centred at origin, lying in the XY plane.
  • Depth axis = Z (the 3D generator convention: front face at min-Z = Z=0 side).
  • Thin depth: 2% of the larger dimension (just enough for OBB to detect depth axis).

Usage:
    python -m object_placement.wall_mounted.generate_window_plane \\
        --output-dir outputs/living_room9

    # Only specific types:
    python -m object_placement.wall_mounted.generate_window_plane \\
        --output-dir outputs/living_room9 --types window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GSA_ROOT  = _REPO_ROOT / "Grounded-Segment-Anything"
sys.path.insert(0, str(_GSA_ROOT / "segment_anything"))

_SAM_CKPT       = _GSA_ROOT / "weights/sam_vit_h_4b8939.pth"
_SAM_MODEL_TYPE = "vit_h"
_BG_GREY        = (185, 185, 185)
_BG_TOL         = 25

_sam_predictor = None

def _get_sam(device: str = "cuda"):
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import SamPredictor, sam_model_registry
        print("[window_plane] Loading SAM …")
        sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=str(_SAM_CKPT))
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def _infer_bg_color(arr):
    """Infer the dominant background colour from a thin border around the image.

    The image-edit model is supposed to put the object on grey (185,185,185),
    but often drifts to cream / off-white for art / paintings.  Reading the
    actual border colour lets the FG threshold work regardless.
    """
    import numpy as np
    H, W = arr.shape[:2]
    t = max(4, min(H, W) // 32)
    border = np.concatenate([
        arr[:t, :].reshape(-1, 3),
        arr[-t:, :].reshape(-1, 3),
        arr[:, :t].reshape(-1, 3),
        arr[:, -t:].reshape(-1, 3),
    ], axis=0)
    return tuple(int(c) for c in np.median(border, axis=0))


def _fg_mask_from_border(arr, tol: int = 25):
    """Return a boolean FG mask using the auto-detected border colour."""
    import numpy as np
    bg = np.array(_infer_bg_color(arr), dtype=np.float32)
    return (np.abs(arr.astype(np.float32) - bg).max(axis=2) > tol)


def _segment_and_crop(img: Image.Image, device: str = "cuda") -> Image.Image:
    """Use SAM to segment the object from its background.  The image-edit model
    usually produces a grey (185,185,185) background but sometimes drifts to
    cream / off-white, so we auto-detect the actual background colour from a
    thin border of the image and fall back to a centre-point SAM prompt when
    the threshold bbox covers nearly the whole image.
    """
    import numpy as np

    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]

    is_fg = _fg_mask_from_border(arr, tol=_BG_TOL)
    rows  = np.where(is_fg.any(axis=1))[0]
    cols  = np.where(is_fg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        print("  [window_plane] no foreground — using full image")
        return img.convert("RGB")

    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    box_frac = ((r1 - r0) * (c1 - c0)) / float(H * W)

    try:
        predictor = _get_sam(device)
        predictor.set_image(arr)
        if box_frac > 0.90:
            # Border threshold failed (bg colour drifts into the painting).
            # Prompt SAM with a centre point — paintings are always near centre.
            print(f"  [window_plane] fg covers {box_frac:.0%} — using centre-point SAM prompt")
            pts = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pts, point_labels=lbl, multimask_output=True)
        else:
            box = np.array([c0, r0, c1, r1], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        mask = masks[int(np.argmax(scores))].astype(bool)   # (H, W)
        print(f"  [window_plane] SAM mask: {mask.sum()} px foreground")
    except Exception as e:
        print(f"  [window_plane] SAM failed ({e}) — falling back to bbox crop")
        return Image.fromarray(arr[r0:r1, c0:c1])

    # Apply mask: set non-mask pixels to grey background
    result = arr.copy()
    result[~mask] = _BG_GREY

    # Crop to mask tight bbox
    m_rows = np.where(mask.any(axis=1))[0]
    m_cols = np.where(mask.any(axis=0))[0]
    if len(m_rows) == 0:
        return Image.fromarray(arr[r0:r1, c0:c1])
    mr0, mr1 = int(m_rows[0]), int(m_rows[-1]) + 1
    mc0, mc1 = int(m_cols[0]), int(m_cols[-1]) + 1
    return Image.fromarray(result[mr0:mr1, mc0:mc1])

import numpy as np
from PIL import Image


def _sam_mask(img_rgb: np.ndarray, device: str = "cuda") -> np.ndarray | None:
    """Run SAM to get a clean binary mask of the object.  Returns None on
    failure so callers can fall back.

    The image-edit output's background colour is detected from a border ring
    rather than assumed to be hard grey (the model sometimes drifts to cream
    / off-white for art).  If the border-threshold FG bbox covers nearly the
    whole image (common for paintings with wash-out backgrounds), we fall back
    to a centre-point SAM prompt — art/paintings are always near image centre.
    """
    H, W = img_rgb.shape[:2]
    is_fg = _fg_mask_from_border(img_rgb, tol=_BG_TOL)
    rows = np.where(is_fg.any(axis=1))[0]
    cols = np.where(is_fg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    box_frac = ((r1 - r0) * (c1 - c0)) / float(H * W)
    try:
        predictor = _get_sam(device)
        predictor.set_image(img_rgb)
        if box_frac > 0.90:
            print(f"  [rectify] fg covers {box_frac:.0%} — centre-point SAM prompt")
            pts = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pts, point_labels=lbl, multimask_output=True)
        else:
            box = np.array([c0, r0, c1, r1], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        mask = masks[int(np.argmax(scores))].astype(bool)
        print(f"  [rectify] SAM mask: {int(mask.sum())} px foreground")
        return mask
    except Exception as e:
        print(f"  [rectify] SAM failed ({e}) — falling back to grey-threshold")
        return None


def _rectify_perspective(img_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Find the tilted rectangle of the object and unwarp to an axis-aligned rect.

    Pipeline:
      1. Run SAM (box-prompted) on the full grey-padded image to get a clean
         binary mask of the window/art.  SAM gives a tight boundary that
         includes interior grey-looking content (sky, reflections, wall wash)
         which grey-thresholding would drop.
      2. Take the largest contour of the mask and compute minAreaRect.
         minAreaRect is robust to small contour noise and always returns 4
         corners of the minimum-area rotated rectangle.
      3. If the tilt is small, return the axis-aligned bbox crop.
      4. Otherwise warpPerspective the tilted rect → axis-aligned rect.
    """
    try:
        import cv2
    except ImportError:
        print("  [rectify] opencv not available — skipping rectification")
        return img_rgb

    H_img, W_img = img_rgb.shape[:2]

    # --- Mask: prefer SAM; fall back to grey-threshold + morph-close ----------
    sam_mask = _sam_mask(img_rgb, device=device)
    if sam_mask is not None:
        mask8 = sam_mask.astype(np.uint8) * 255
        # SAM boundary softens by 1-2 px at corners, often dipping into the
        # surrounding grey padding.  A small erosion pulls the boundary
        # strictly INSIDE the true window so the fitted quad doesn't include
        # grey pixels at its corners, which would show up as grey borders
        # after warpPerspective.
        _sam_erode = 5
        sam_kernel = np.ones((_sam_erode, _sam_erode), np.uint8)
        mask8 = cv2.erode(mask8, sam_kernel, iterations=1)
        ks = 0  # no morph-close used
    else:
        grey  = np.array(_BG_GREY, dtype=np.float32)
        is_fg = (np.abs(img_rgb.astype(np.float32) - grey) > _BG_TOL).any(axis=2)
        mask8 = is_fg.astype(np.uint8) * 255
        # Close interior holes (glass panes / mullions matching grey bg).
        ks = max(15, min(H_img, W_img) // 20)
        if ks % 2 == 0:
            ks += 1
        kernel = np.ones((ks, ks), np.uint8)
        mask8  = cv2.morphologyEx(mask8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    largest = max(contours, key=cv2.contourArea)

    # --- Fit a 4-corner quad --------------------------------------------------
    # A perspective-distorted window is a TRAPEZOID, not a rotated rectangle.
    # minAreaRect can only model rotation, so it collapses a trapezoid into a
    # near-axis-aligned rect and discards the perspective foreshortening we
    # want to undo.  On a clean SAM mask, approxPolyDP on the convex hull
    # robustly recovers the 4 trapezoid corners.  Fall back to minAreaRect
    # only if that fails (unlikely with SAM; common with noisy grey-thresh).
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    box  = None
    for _eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, _eps * peri, True)
        if len(approx) == 4:
            box = approx.reshape(-1, 2).astype(np.float32)
            break
    if box is None:
        rect = cv2.minAreaRect(largest)
        (_cx, _cy), (_rw, _rh), _ang = rect
        if min(_rw, _rh) < 10:
            return img_rgb
        # Grey-threshold fallback path: closed mask dilates ~ks/2 px beyond
        # the true edge, so shrink the fitted rect inward.
        if sam_mask is None and ks > 0:
            _margin = ks / 2.0 + 1.0
            _rw = max(10.0, _rw - 2 * _margin)
            _rh = max(10.0, _rh - 2 * _margin)
            rect = ((_cx, _cy), (_rw, _rh), _ang)
        box = cv2.boxPoints(rect).astype(np.float32)
        print(f"  [rectify] approxPolyDP failed — using minAreaRect")

    # Order corners: TL, TR, BR, BL (works for rotations < ~45°)
    s    = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Measure edge tilt for the early-out.  minAreaRect is stable so a tiny
    # tilt here is a real (small) rotation, not contour jitter.
    def _edge_angle_deg(p_from, p_to):
        dx, dy = (p_to - p_from)
        return float(np.degrees(np.arctan2(abs(dy), abs(dx))))   # 0 = H, 90 = V
    ang_top   = _edge_angle_deg(tl, tr)
    ang_bot   = _edge_angle_deg(bl, br)
    ang_left  = 90.0 - _edge_angle_deg(tl, bl)
    ang_right = 90.0 - _edge_angle_deg(tr, br)
    max_tilt  = max(ang_top, ang_bot, ang_left, ang_right)

    _src = "SAM" if sam_mask is not None else "grey-thresh"

    if max_tilt < 1.5:
        x, y, w, h = cv2.boundingRect(largest)
        print(f"  [rectify] already axis-aligned (max edge tilt {max_tilt:.2f}°) "
              f"— bbox crop {w}×{h}  (mask={_src})")
        return img_rgb[y:y+h, x:x+w]

    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if W < 10 or H < 10:
        return img_rgb

    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(corners, dst)
    out = cv2.warpPerspective(img_rgb, M, (W, H), borderValue=_BG_GREY)

    print(f"  [rectify] minAreaRect tilt {max_tilt:.2f}° — warped "
          f"{img_rgb.shape[1]}×{img_rgb.shape[0]} → {W}×{H}  (mask={_src})")
    return out


# ── GLB writing ───────────────────────────────────────────────────────────────
# Dense vertex grid with vertex colors baked from the texture.
# Using a grid (not just 4 corners) gives the software vertex-color rasterizer
# in place_objects._overlay_glbs enough samples to reproduce the texture detail.
# Single-buffer GLB (geometry + texture in one BIN chunk) for trimesh compat.

import struct
import io

_MAX_GRID = 256   # max grid cells per dimension


def _pack_glb(vertices: np.ndarray,
              faces:    np.ndarray,
              uvs:      np.ndarray,
              colors:   np.ndarray,
              texture_img: Image.Image) -> bytes:
    """Pack a single-buffer GLB with POSITION, TEXCOORD_0, COLOR_0, and indices.

    vertices: (N,3) float32
    faces:    (F,3) uint32
    uvs:      (N,2) float32  V flipped (OpenGL: V=0 at bottom)
    colors:   (N,3) uint8    vertex colors sampled from texture
    texture_img: PIL Image   embedded as PNG for UV-capable renderers
    """
    import json as _json

    def _pad4(n: int) -> int:
        return (4 - n % 4) % 4

    # ── Encode all binary blobs ───────────────────────────────────────────────
    pos_raw = vertices.astype(np.float32).tobytes()
    uv_raw  = uvs.astype(np.float32).tobytes()
    # COLOR_0: VEC4 UNSIGNED_BYTE normalized (trimesh expects alpha channel)
    # Pad RGB to RGBA so any loader handles it; alpha = 255
    rgba    = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], axis=1)
    col_raw = rgba.tobytes()
    idx_raw = faces.astype(np.uint32).tobytes()

    buf_tex = io.BytesIO()
    texture_img.save(buf_tex, format="PNG")
    tex_raw = buf_tex.getvalue()
    tex_len = len(tex_raw)   # actual texture byte length (before padding)

    # Pad each section to 4-byte boundary
    def _pad(b: bytes) -> bytes:
        return b + b"\x00" * _pad4(len(b))

    pos_bytes = _pad(pos_raw);  pos_len = len(pos_raw)
    uv_bytes  = _pad(uv_raw);   uv_len  = len(uv_raw)
    col_bytes = _pad(col_raw);  col_len = len(col_raw)
    idx_bytes = _pad(idx_raw);  idx_len = len(idx_raw)
    tex_bytes = _pad(tex_raw)

    pos_offset = 0
    uv_offset  = len(pos_bytes)
    col_offset = uv_offset  + len(uv_bytes)
    idx_offset = col_offset + len(col_bytes)
    tex_offset = idx_offset + len(idx_bytes)

    # Single BIN buffer: all data concatenated
    bin_buf = pos_bytes + uv_bytes + col_bytes + idx_bytes + tex_bytes
    bin_len = len(bin_buf)

    N = len(vertices)
    F = len(faces)
    pos_min = [float(x) for x in vertices.min(axis=0)]
    pos_max = [float(x) for x in vertices.max(axis=0)]

    # ── glTF JSON ─────────────────────────────────────────────────────────────
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes":  [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION":   0,
                    "TEXCOORD_0": 1,
                    "COLOR_0":    2,
                },
                "indices":  3,
                "material": 0,
            }]
        }],
        "accessors": [
            {   # 0 POSITION
                "bufferView": 0, "byteOffset": 0,
                "componentType": 5126, "count": N, "type": "VEC3",
                "min": pos_min, "max": pos_max,
            },
            {   # 1 TEXCOORD_0
                "bufferView": 1, "byteOffset": 0,
                "componentType": 5126, "count": N, "type": "VEC2",
            },
            {   # 2 COLOR_0 — VEC4 UNSIGNED_BYTE normalized
                "bufferView": 2, "byteOffset": 0,
                "componentType": 5121, "count": N, "type": "VEC4",
                "normalized": True,
            },
            {   # 3 indices
                "bufferView": 3, "byteOffset": 0,
                "componentType": 5125, "count": F * 3, "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset,  "byteLength": uv_len,  "target": 34962},
            {"buffer": 0, "byteOffset": col_offset,  "byteLength": col_len,  "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963},
            {"buffer": 0, "byteOffset": tex_offset,  "byteLength": tex_len},  # texture image
        ],
        "buffers": [{"byteLength": bin_len}],
        "images":   [{"bufferView": 4, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        }],
    }

    json_bytes = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * _pad4(len(json_bytes))

    def _chunk(ctype: int, data: bytes) -> bytes:
        return struct.pack("<II", len(data), ctype) + data

    body      = _chunk(0x4E4F534A, json_bytes) + _chunk(0x004E4942, bin_buf)
    total_len = 12 + len(body)
    header    = struct.pack("<III", 0x46546C67, 2, total_len)
    return header + body


def make_window_plane(img: Image.Image, depth_frac: float = 0.005,
                      device: str = "cuda", rectify: bool = False) -> bytes:
    """Create a thin flat plane GLB textured with the (SAM-segmented) image.

    Uses a dense vertex grid (up to _MAX_GRID × _MAX_GRID, matching texture pixel
    count) so that vertex colors baked from the texture give the place_objects
    overlay renderer pixel-accurate coverage without UV-texture rasterisation.

    Plane is centred at origin, lying in the XY plane:
      width  = 2.0  (along X)
      height = 2.0 * (H/W)  (along Y)
      depth  = depth_frac * max(w,h)  (along Z — just enough for OBB detection)

    rectify=True: after SAM segmentation, apply perspective-unwarp to
    produce a fully frontal rectangular crop (useful for paintings/art
    captured at an angle — removes trapezoidal distortion).
    """
    # Pipeline:
    #   rectify=True: run _rectify_perspective on the FULL grey-padded image.
    #     It finds the tilted rectangle via minAreaRect on the FG mask, then
    #     warpPerspectives it to an upright axis-aligned rectangle.  We must
    #     operate on the un-cropped image here — a pre-tightened bbox crop has
    #     FG touching all 4 edges, so the morph-close would fill the tilted
    #     corner triangles and make the minAreaRect degenerate to 0° tilt.
    #     Rectify already returns a tight crop (via bbox or warp), so no SAM
    #     pass is needed afterwards.
    #   rectify=False: use SAM (with grey-threshold fallback) to segment + crop.
    img_rgb_full = img.convert("RGB")
    if rectify:
        img_rgb = Image.fromarray(
            _rectify_perspective(np.array(img_rgb_full), device=device)
        )
    else:
        img_rgb = _segment_and_crop(img_rgb_full, device=device)
    W_px, H_px = img_rgb.size
    img_arr   = np.array(img_rgb, dtype=np.uint8)
    print(f"  [window_plane] texture size after "
          f"{'rectify' if rectify else 'SAM crop'}: {W_px}×{H_px}"
          f"  aspect={H_px/W_px:.3f}")

    aspect = H_px / W_px
    half_w = 1.0
    half_h = aspect
    depth  = max(half_w, half_h) * depth_frac   # very thin; just enough for OBB axis detection

    # Grid resolution = texture pixels capped at _MAX_GRID for each dimension.
    # This gives ~1 vertex per texture pixel, so vertex-color sampling is
    # pixel-accurate rather than bilinear-smeared.
    Gw = min(_MAX_GRID, W_px)
    Gh = min(_MAX_GRID, H_px)
    n_grid = (Gw + 1) * (Gh + 1)

    # Grid UV: u 0→1 left→right, v 0→1 top→bottom (image convention)
    us = np.linspace(0.0, 1.0, Gw + 1, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, Gh + 1, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)   # (Gh+1, Gw+1); vv=0 at top, vv=1 at bottom

    # World XY from UV
    xs = (uu * 2.0 - 1.0) * half_w                    # -half_w → +half_w
    ys = (1.0 - vv) * 2.0 * half_h - half_h           # +half_h (top) → -half_h (bottom)

    # Single-layer plane at z=0 (no back face — avoids double-mesh in OBJ viewer).
    # A tiny depth offset on 4 corner verts is enough for OBB to detect Z as depth.
    verts = np.stack([xs.ravel(), ys.ravel(),
                      np.zeros(n_grid, dtype=np.float32)], axis=1)

    # OpenGL UV: V flipped (V=0 at bottom of image)
    uvs = np.stack([uu.ravel(), 1.0 - vv.ravel()], axis=1).astype(np.float32)

    # Bake vertex colors by sampling texture at each UV
    tx = np.clip((uu.ravel() * (W_px - 1)).astype(np.int32), 0, W_px - 1)
    ty = np.clip((vv.ravel() * (H_px - 1)).astype(np.int32), 0, H_px - 1)
    colors = img_arr[ty, tx].astype(np.uint8)  # (n_grid, 3)

    # Build quad→triangle faces (single-sided, CCW from -Z)
    face_list = []
    for row in range(Gh):
        for col in range(Gw):
            i0 = row * (Gw + 1) + col
            i1 = i0 + 1
            i2 = i0 + (Gw + 1) + 1
            i3 = i0 + (Gw + 1)
            face_list.append([i0, i1, i2]); face_list.append([i0, i2, i3])
    faces = np.array(face_list, dtype=np.uint32)

    return _pack_glb(verts, faces, uvs, colors, img_rgb)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(output_dir: str | Path,
        types: list[str] | None = None,
        force: bool = False,
        device: str = "cuda") -> None:
    out_dir     = Path(output_dir) / "wall_mounted"
    inpaint_dir = out_dir / "inpainted"
    obj_dir     = out_dir / "objects"
    results_p   = out_dir / "segment_results.json"

    obj_dir.mkdir(parents=True, exist_ok=True)

    with open(results_p) as f:
        data = json.load(f)

    target_types = set(types) if types else {"window", "door"}

    for seg in data.get("segments", []):
        idx      = seg.get("index")
        obj_type = seg.get("type", "")
        if obj_type not in target_types:
            continue

        inpaint_file = seg.get("inpaint_file")
        if not inpaint_file:
            print(f"[window_plane] seg {idx:02d}: no inpaint_file — skipping")
            continue

        src = inpaint_dir / inpaint_file
        if not src.exists():
            print(f"[window_plane] {src.name} not found — skipping")
            continue

        glb_name = f"inpaint_{idx:02d}_{obj_type}.glb"
        glb_path = obj_dir / glb_name

        if glb_path.exists() and not force:
            print(f"[window_plane] {glb_name} exists — skipping (use --force to regenerate)")
            seg["glb_file"] = glb_name
            continue

        print(f"[window_plane] {idx:02d} {obj_type} → {glb_name}")
        img     = Image.open(src).convert("RGB")
        # Rectify every plane type — windows/doors are often captured at a
        # slight tilt, and art/painting/frame need the same fronto-parallel
        # correction.  Mirrors do NOT go through this code path — they use
        # the the 3D generator 3D reconstruction in object_generation.py.
        glb_bytes = make_window_plane(img, device=device, rectify=True)
        glb_path.write_bytes(glb_bytes)
        seg["glb_file"] = glb_name
        print(f"  saved → {glb_path}")

    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)
    print("\n[window_plane] Done. segment_results.json updated.")


def main():
    ap = argparse.ArgumentParser(
        description="Generate flat plane GLBs textured with inpainted window images."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output directory (contains wall_mounted/)")
    ap.add_argument("--types", default="window,door",
                    help="Comma-separated object types (default: window,door)")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if GLB already exists")
    ap.add_argument("--device", default="cuda",
                    help="Device for SAM (cuda or cpu)")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",")]
    run(output_dir=args.output_dir, types=types, force=args.force, device=args.device)


if __name__ == "__main__":
    main()
