"""
floorplan/vggt_estimates/manhattan.py
---------------------------
Manhattan-world room estimation from VGGT outputs.

Finds the 6 bounding planes of a rectangular room (floor, ceiling, 4 walls),
computes the 8 corners and 12 edges that form the room box, then projects
them onto the reference image.

Usage
-----
    python -m floorplan.vggt_estimates.manhattan \\
        --vggt-out outputs/vggt_office5 \\
        --image    data/indoor_images/office5.jpg

    from floorplan.vggt_estimates.manhattan import estimate_and_visualize
    estimate_and_visualize("outputs/vggt_office5", "data/indoor_images/office5.jpg")
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_outputs(vggt_out_dir: str, frame: int = 0) -> dict:
    d = Path(vggt_out_dir)

    cam_path    = d / "camera.json"
    normal_path = d / f"normal_{frame}.npy"
    depth_path  = d / f"depth_{frame}.npy"

    required = [cam_path, normal_path, depth_path]
    missing  = [str(p.name) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "VGGT output directory is missing required files: "
            f"{missing}. Expected a raw VGGT output folder such as outputs/vggt_office5, "
            "not the main HARMONY walls output folder."
        )

    cam_json = json.loads(cam_path.read_text())
    if isinstance(cam_json, list):
        if not (0 <= frame < len(cam_json)):
            raise IndexError(f"Frame {frame} out of range for {cam_path} ({len(cam_json)} frame(s))")
        cam = cam_json[frame]
    elif isinstance(cam_json, dict):
        if "intrinsic_3x3" not in cam_json or "extrinsic_3x4" not in cam_json:
            raise ValueError(
                f"{cam_path} is a dict, but it does not look like a raw VGGT camera export. "
                "It is missing intrinsic_3x3/extrinsic_3x4."
            )
        cam = cam_json
    else:
        raise TypeError(f"Unsupported camera.json format in {cam_path}: {type(cam_json).__name__}")

    K = np.array(cam["intrinsic_3x3"], dtype=np.float64)
    E = np.array(cam["extrinsic_3x4"], dtype=np.float64)
    R = E[:3, :3]

    # VGGT depth is metric (metres); normals are full-resolution.
    depth   = np.load(str(depth_path)).astype(np.float32)
    normals = np.load(str(normal_path)).astype(np.float32)

    return dict(
        K=K, R=R, t=E[:3, 3],
        depth=depth, normals=normals,
        H_orig=cam["image_size_hw_orig"][0],
        W_orig=cam["image_size_hw_orig"][1],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3-D geometry
# ─────────────────────────────────────────────────────────────────────────────

def _unproject(depth: np.ndarray, K: np.ndarray,
               R: np.ndarray, t: np.ndarray,
               orig_H: int, orig_W: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth → world-space points.

    `depth` may be a subsampled view of the original (H_orig × W_orig) map.
    orig_H / orig_W are the original dimensions so we can recover correct
    pixel coordinates even when depth is smaller than the full image.

    Returns (pts_w (N,3), flat valid mask).
    """
    H, W   = depth.shape
    # pixel stride in the original image (1.0 if not subsampled)
    su     = orig_W / W
    sv     = orig_H / H
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # original-image pixel coordinates of each subsampled cell
    u, v = np.meshgrid(
        (np.arange(W, dtype=np.float64) + 0.5) * su - 0.5,  # centre of each bin
        (np.arange(H, dtype=np.float64) + 0.5) * sv - 0.5,
    )
    valid = depth > 0
    z     = depth[valid].astype(np.float64)
    pts_c = np.stack([(u[valid] - cx) / fx * z,
                      (v[valid] - cy) / fy * z,
                      z], axis=1)                  # (N, 3) camera space
    pts_w = (R.T @ (pts_c - t).T).T               # (N, 3) world space
    return pts_w, valid


def _project_pts(pts_w: np.ndarray, K: np.ndarray,
                 R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project (N,3) world → (N,2) pixel coords."""
    pts_c = (R @ pts_w.T).T + t
    eps   = 1e-6
    pts_c[:, 2] = np.maximum(pts_c[:, 2], eps)
    h = (K @ pts_c.T).T
    return h[:, :2] / h[:, 2:3]


# ─────────────────────────────────────────────────────────────────────────────
# Manhattan axis estimation
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_manhattan_axes(normals: np.ndarray,
                              n_samples: int = 150_000) -> np.ndarray:
    """
    Estimate 3 orthogonal Manhattan axes from unit normals (N,3).
    Returns (3,3) – each row is one axis.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(normals), min(n_samples, len(normals)), replace=False)
    n   = normals[idx]

    axes = np.eye(3, dtype=np.float64)
    for _ in range(8):
        dots   = np.abs(n @ axes.T)              # (N,3)
        labels = dots.argmax(axis=1)
        new_axes = []
        for k in range(3):
            grp = n[labels == k]
            if len(grp) < 20:
                new_axes.append(axes[k])
                continue
            signs      = np.sign(grp @ axes[k])
            signs[signs == 0] = 1
            _, _, Vt   = np.linalg.svd(grp * signs[:, None], full_matrices=False)
            new_axes.append(Vt[0])
        M = np.stack(new_axes)
        U, _, Vt2 = np.linalg.svd(M)
        axes = U @ Vt2
        if np.linalg.det(axes) < 0:
            axes[2] *= -1
    return axes.astype(np.float64)


def _vertical_axis_idx(axes: np.ndarray) -> int:
    """Return the index of the axis most aligned with world-Y (up/down)."""
    return int(np.abs(axes[:, 1]).argmax())


# ─────────────────────────────────────────────────────────────────────────────
# Room boundary plane detection
# ─────────────────────────────────────────────────────────────────────────────

def _find_two_walls(projections: np.ndarray,
                    pct_inner: float = 5.0,
                    n_bins: int = 200) -> tuple[float, float]:
    """
    Given the 1-D projection of all room points onto a Manhattan axis, find
    the two room-bounding planes (min-side and max-side walls).

    Uses histogram peak-finding near the low and high tails.
    Returns (d_low, d_high) – the two plane offsets.
    """
    lo = np.percentile(projections, pct_inner)
    hi = np.percentile(projections, 100 - pct_inner)
    rng   = hi - lo
    lo_hi = lo + rng * 0.35
    hi_lo = hi - rng * 0.35

    counts_lo, edges_lo = np.histogram(projections[projections <= lo_hi], bins=n_bins)
    counts_hi, edges_hi = np.histogram(projections[projections >= hi_lo], bins=n_bins)

    def _peak_centre(counts, edges):
        i = int(np.argmax(counts))
        return float(0.5 * (edges[i] + edges[i + 1]))

    return _peak_centre(counts_lo, edges_lo), _peak_centre(counts_hi, edges_hi)


# ─────────────────────────────────────────────────────────────────────────────
# Box corners and edges
# ─────────────────────────────────────────────────────────────────────────────

def _plane_intersection_point(normals_3: np.ndarray,
                               offsets_3: np.ndarray) -> np.ndarray:
    """
    Solve the 3-plane intersection: [n0;n1;n2] @ x = [d0;d1;d2].
    Returns the corner point (3,).
    """
    x, *_ = np.linalg.lstsq(normals_3, offsets_3, rcond=None)
    return x


def _build_box(axes: np.ndarray, planes: dict) -> tuple[np.ndarray, list]:
    """
    Build the 8 corners and 12 edges of the room box.

    Corner bit-coding: bit k → which side (0=low, 1=high) of axis-k plane.

    Returns:
        corners – (8,3) world-space corner points
        edges   – list of (i, j, ax, face_ax1, face_side1, face_ax2, face_side2)
                  where ax = axis the edge runs along,
                  (face_ax1, face_side1) and (face_ax2, face_side2) are the
                  two faces that share this edge.
    """
    corners = np.zeros((8, 3), dtype=np.float64)
    for bits in range(8):
        sides    = [(bits >> k) & 1 for k in range(3)]
        corners[bits] = _plane_intersection_point(
            axes[[0, 1, 2]],
            np.array([planes[k][s] for k, s in enumerate(sides)])
        )

    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = i ^ j
            if diff and (diff & (diff - 1)) == 0:
                ax = int(np.log2(diff))
                edges.append((i, j, ax))

    return corners, edges


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def estimate_manhattan(vggt_out_dir: str, frame: int = 0,
                       subsample: int = 4) -> dict:
    """
    Estimate the Manhattan room box from VGGT outputs
    (camera.json + metric depth_{frame}.npy + normal_{frame}.npy).

    Returns dict with keys: axes, planes, corners, edges, K, R, t, H, W
    """
    data   = _load_outputs(vggt_out_dir, frame)
    K, R, t        = data["K"], data["R"], data["t"]
    H_orig, W_orig = data["H_orig"], data["W_orig"]

    depth_sub   = data["depth"][::subsample, ::subsample]
    normals_sub = data["normals"][::subsample, ::subsample]

    pts_w, valid = _unproject(depth_sub, K, R, t, H_orig, W_orig)
    norms_flat   = normals_sub[valid].astype(np.float64)
    mag          = np.linalg.norm(norms_flat, axis=1, keepdims=True)
    norms_flat  /= np.maximum(mag, 1e-8)

    print(f"[manhattan] {len(pts_w):,} valid points  (subsample ×{subsample})")

    # ── Manhattan axes ─────────────────────────────────────────────────────
    axes  = _estimate_manhattan_axes(norms_flat)
    v_idx = _vertical_axis_idx(axes)

    # orient vertical axis so it points "up" in world coords (+Y = up means
    # axis_v should have a negative Y component for OpenCV convention where Y↓)
    if axes[v_idx, 1] > 0:
        axes[v_idx] *= -1

    print(f"[manhattan] Axes (rows):\n"
          f"  ax{v_idx} (vertical): {np.round(axes[v_idx], 3)}\n"
          f"  ax{[i for i in range(3) if i != v_idx][0]}: "
          f"{np.round(axes[[i for i in range(3) if i != v_idx][0]], 3)}\n"
          f"  ax{[i for i in range(3) if i != v_idx][1]}: "
          f"{np.round(axes[[i for i in range(3) if i != v_idx][1]], 3)}")

    # ── Pass 1: histogram-peak approximate layout ─────────────────────────────
    # Fast, dense-cluster based estimate — gives approximate wall positions
    # sufficient to define the Manhattan coordinate system and room extent.
    # Furniture clusters (shelves, counters) will pull planes inward, so this
    # is only used as a reference frame for Pass 2.
    print("[manhattan] Pass 1 — histogram-peak approximate layout")
    planes_init: dict = {}
    labels = {v_idx: "vertical", **{i: f"horiz{n}" for n, i in
                                     enumerate(i for i in range(3) if i != v_idx)}}
    for k in range(3):
        proj           = pts_w @ axes[k]
        d_lo, d_hi     = _find_two_walls(proj)
        planes_init[k] = (d_lo, d_hi)
        print(f"[manhattan]   Axis {k} ({labels[k]}): d={d_lo:.4f}  {d_hi:.4f}")

    corners_init, edges_init = _build_box(axes, planes_init)

    # ── Pass 2: vertical extension from image-row floor/ceiling pixels ─────────
    # Floor tiles are always in the BOTTOM image rows; ceiling in the TOP rows.
    # Unproject those pixel bands directly from the full-resolution depth map
    # — every floor/ceiling pixel contributes, bypassing the subsampled cloud
    # and the shelf/counter surfaces that dominate point-cloud statistics.
    depth_full_f32 = data["depth"].astype(np.float32)
    H_f, W_f       = depth_full_f32.shape
    # Scale intrinsics to full-res depth dimensions
    sx = W_f / W_orig;  sy = H_f / H_orig
    fx_f = float(K[0, 0]) * sx;  fy_f = float(K[1, 1]) * sy
    cx_f = float(K[0, 2]) * sx;  cy_f = float(K[1, 2]) * sy

    d_lo_i, d_hi_i = planes_init[v_idx]
    span_v         = max(d_hi_i - d_lo_i, 1e-6)
    # Allow floor to extend up to 1.5× the room span below the Pass-1 estimate.
    # A shelf/counter in the back corner hides the actual floor tiles and causes
    # Pass-1 to anchor the floor at counter-top level.  The bottom image rows
    # always contain actual floor tiles (below the camera, unoccluded), so
    # using their p5 projection value with a generous cap finds the real floor.
    floor_ext_cap  = span_v * 1.5
    row_frac       = 0.20   # bottom/top 20 % of rows
    min_pts        = 50

    def _row_band_proj_v(r0, r1):
        band  = depth_full_f32[r0:r1, :]
        valid = band > 0
        if valid.sum() < min_pts:
            return None
        rows, cols = np.where(valid)
        z = band[valid].astype(np.float64)
        pts_c = np.stack([(cols       - cx_f) / fx_f * z,
                          (rows + r0  - cy_f) / fy_f * z,
                          z], axis=1)
        pts_w_band = (R.T @ (pts_c - t).T).T
        return (pts_w_band @ axes[v_idx]).astype(np.float32)

    # floor — bottom row band: p5 of those pixels, capped at 1.5× room span
    pv_floor = _row_band_proj_v(int(H_f * (1 - row_frac)), H_f)
    if pv_floor is not None and len(pv_floor) >= min_pts:
        d_lo = float(np.percentile(pv_floor, 5))
        d_lo = max(d_lo, d_lo_i - floor_ext_cap)
        d_lo = min(d_lo, d_lo_i)           # only extend outward (lower)
    else:
        d_lo = d_lo_i

    # ceiling — keep Pass-1 estimate; the ceiling is typically correct from the
    # histogram peak and extending it with top-row pixels tends to overshoot.
    d_hi = d_hi_i

    print(f"[manhattan] Pass 2 floor extension: "
          f"floor {d_lo_i:.4f}→{d_lo:.4f}  ceil unchanged {d_hi_i:.4f}  "
          f"floor_cap -{floor_ext_cap:.4f}")

    planes         = dict(planes_init)
    planes[v_idx]  = (d_lo, d_hi)
    corners, edges = _build_box(axes, planes)

    # ── Deepest corner via camera-space Z of box corners ─────────────────────
    # Use the 3D cam-Z of each floor corner (already in camera coordinates).
    # This is the ground-truth depth of the box vertex from the camera — it
    # doesn't depend on depth-map sampling and is unaffected by windows or
    # furniture. The floor corner with the largest cam-Z is the back-wall corner
    # farthest from the camera, i.e. the alignment anchor we want.
    # Only floor corners (bit v_idx == 0) are considered; ceiling corners may
    # have higher cam-Z but they are not the alignment anchor.
    fx, fy     = K[0, 0], K[1, 1]
    cx, cy     = K[0, 2], K[1, 2]
    corners_c  = (R @ corners.T).T + t

    cam_z_vals         = np.full(8, -1.0)
    deepest_corner_idx = 0
    for ci in range(8):
        if (ci >> v_idx) & 1:          # skip ceiling corners
            continue
        c = corners_c[ci]
        if c[2] <= 0:
            continue
        u_p = c[0] / c[2] * fx + cx
        v_p = c[1] / c[2] * fy + cy
        if not (0 <= u_p < W_orig and 0 <= v_p < H_orig):
            continue
        cam_z_vals[ci] = float(c[2])
    deepest_corner_idx = int(np.argmax(cam_z_vals))
    print(f"[manhattan] Floor corner cam-Z: {np.round(cam_z_vals, 3)}"
          f"  → deepest idx={deepest_corner_idx}"
          f"  world={np.round(corners[deepest_corner_idx], 3)}")
    print(f"[manhattan] Box: 8 corners, {len(edges)} edges")

    return dict(axes=axes, v_idx=v_idx, planes=planes,
                corners=corners, edges=edges,
                corners_pass1=corners_init, edges_pass1=edges_init,
                corners_init=corners_init, edges_init=edges_init,
                deepest_corner_idx=deepest_corner_idx,
                K=K, R=R, t=t, H=H_orig, W=W_orig)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# colours per Manhattan axis index
_AXIS_COLOUR = [
    (255,  80,  80),   # axis 0 – red
    ( 80, 220,  80),   # axis 1 – green
    ( 80, 140, 255),   # axis 2 – blue
]
_CORNER_COLOUR = (255, 230, 50)


def _draw_box_overlay(image_path: str, corners: np.ndarray, edges: list,
                      axes: np.ndarray, v_idx: int,
                      K: np.ndarray, R: np.ndarray, t: np.ndarray,
                      H: int, W: int,
                      out_path: str,
                      label: str = "",
                      max_display_px: int = 1920,
                      line_width: int = 3,
                      corner_radius: int = 6,
                      deepest_corner_idx: int | None = None) -> str:
    """Draw a Manhattan box wireframe overlay on the reference image and save."""
    img   = Image.open(image_path).convert("RGB")
    scale = min(1.0, max_display_px / max(W, H))
    disp_W, disp_H = int(W * scale), int(H * scale)
    if scale < 1.0:
        img = img.resize((disp_W, disp_H), Image.LANCZOS)

    corners_px   = _project_pts(corners, K, R, t) * scale
    corners_c    = (R @ corners.T).T + t
    in_front     = corners_c[:, 2] > 0
    in_frame     = (
        in_front
        & (corners_px[:, 0] >= 0) & (corners_px[:, 0] < disp_W)
        & (corners_px[:, 1] >= 0) & (corners_px[:, 1] < disp_H)
    )

    draw = ImageDraw.Draw(img)
    for i, j, ax in edges:
        if not (in_frame[i] or in_frame[j]) or not (in_front[i] or in_front[j]):
            continue
        x0, y0 = corners_px[i]
        x1, y1 = corners_px[j]
        draw.line([(x0, y0), (x1, y1)], fill=_AXIS_COLOUR[ax % 3], width=line_width)

    for k in range(8):
        if not in_frame[k]:
            continue
        x, y = corners_px[k]
        r    = corner_radius
        # highlight deepest corner in a distinct colour
        fill = (255, 80, 255) if k == deepest_corner_idx else _CORNER_COLOUR
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=fill,
                     outline=(0, 0, 0), width=1)

    h_idxs   = [i for i in range(3) if i != v_idx]
    ax_names = {v_idx: "vertical (floor↔ceil)",
                **{h: f"horiz-{n+1} (walls)" for n, h in enumerate(h_idxs)}}
    lx, ly = 16, 16
    if label:
        draw.text((lx, ly), label, fill=(255, 255, 100))
        ly += 22
    for ax_i, name in ax_names.items():
        col = _AXIS_COLOUR[ax_i % 3]
        draw.rectangle([lx, ly, lx + 28, ly + 16], fill=col)
        draw.text((lx + 34, ly + 2), name, fill=(255, 255, 255))
        ly += 24

    img.save(out_path)
    print(f"[manhattan] Saved → {out_path}")
    return out_path


def visualize_manhattan(vggt_out_dir: str,
                        image_path: str,
                        out_path: str | None = None,
                        frame: int = 0,
                        subsample: int = 4,
                        max_display_px: int = 1920,
                        line_width: int = 3,
                        corner_radius: int = 6) -> str:
    """
    Run Manhattan estimation and save two reference-image overlays:

      manhattan_pass1.png — Pass 1: back-corner anchor box (vertical extent
                            at the deepest back wall, magenta dot = deepest corner).
      manhattan_pass2.png — Pass 2: full-cloud extension of the Pass 1 box.

    Also saves the legacy ``manhattan.png`` (= Pass 2) for callers that expect it.
    Returns path to the Pass 2 overlay.
    """
    result = estimate_manhattan(vggt_out_dir, frame=frame, subsample=subsample)
    K, R, t  = result["K"], result["R"], result["t"]
    H, W     = result["H"], result["W"]
    axes     = result["axes"]
    v_idx    = result["v_idx"]
    deepest  = result.get("deepest_corner_idx")

    base     = Path(out_path).parent if out_path else Path(vggt_out_dir)
    kw = dict(K=K, R=R, t=t, H=H, W=W, axes=axes, v_idx=v_idx,
              image_path=image_path, max_display_px=max_display_px,
              line_width=line_width, corner_radius=corner_radius,
              deepest_corner_idx=deepest)

    # Pass 1 — back-corner anchor
    _draw_box_overlay(
        corners=result["corners_pass1"], edges=result["edges_pass1"],
        out_path=str(base / "manhattan_pass1.png"),
        label="Pass 1 — back-corner vertical anchor",
        **kw)

    # Pass 2 — full-cloud extension
    dest = out_path or str(base / "manhattan_pass2.png")
    _draw_box_overlay(
        corners=result["corners"], edges=result["edges"],
        out_path=dest,
        label="Pass 2 — full-cloud vertical extension",
        **kw)

    # Legacy alias so existing callers still find manhattan.png
    legacy = str(base / "manhattan.png")
    if dest != legacy:
        import shutil as _shutil
        _shutil.copy2(dest, legacy)

    return dest


def estimate_and_visualize(vggt_out_dir: str, image_path: str, **kwargs) -> str:
    return visualize_manhattan(vggt_out_dir, image_path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manhattan room box visualisation")
    ap.add_argument("--vggt-out",  required=True)
    ap.add_argument("--image",     required=True)
    ap.add_argument("--out",       default=None)
    ap.add_argument("--frame",     type=int,   default=0)
    ap.add_argument("--subsample", type=int,   default=4)
    ap.add_argument("--max-px",    type=int,   default=1920,
                    help="Longest side of output image (default 1920)")
    ap.add_argument("--line-width",    type=int, default=3)
    ap.add_argument("--corner-radius", type=int, default=6)
    args = ap.parse_args()

    visualize_manhattan(
        vggt_out_dir=args.vggt_out,
        image_path=args.image,
        out_path=args.out,
        frame=args.frame,
        subsample=args.subsample,
        max_display_px=args.max_px,
        line_width=args.line_width,
        corner_radius=args.corner_radius,
    )
