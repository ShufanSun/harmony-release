"""
floorplan/vggt_estimates/align_to_walls.py
------------------------------------------
Align the VLM camera to the VGGT Manhattan box via an iterative VLM loop.

Pipeline
--------
1. Run Manhattan estimation on VGGT depth + normals → 12-edge orthographic
   room box in VGGT world space.
2. Identify the back-center vertical edge as the alignment anchor:
     floor anchor  = lower image endpoint of the deepest visible vertical edge
     ceiling anchor = upper endpoint of the same edge
3. Produce two static visualizations:
     manhattan_reference.png  — box + anchors on a blank white background
     manhattan_vggt.png       — box + anchors overlaid on the reference photo
4. Iteratively call a VLM to refine the render camera:
     a. Render the room with the current camera.
     b. Overlay the Manhattan box wireframe + anchor markers on that render.
     c. Feed (manhattan_reference.png, render_overlay_{i}.png) to the VLM.
     d. VLM returns yaw / pitch / hfov deltas.
     e. Apply deltas; repeat for n_iter iterations.
5. Return the final camera dict.

Coordinate conventions
----------------------
  VGGT world : OpenCV  (X right, Y down, Z into scene, camera at origin)
  walls.obj  : Y-up CG (X right, Y up,  Z back→camera; back wall at Z=0)

Usage
-----
    from floorplan.vggt_estimates.align_to_walls import align_camera, save_camera
    cam = align_camera("outputs/vggt_office5", "outputs/office5_pipeline")
    save_camera(cam, "outputs/office5_pipeline/camera_vggt.json")

    python -m floorplan.vggt_estimates.align_to_walls \\
        --vggt-out  outputs/vggt_office5 \\
        --walls-out outputs/office5_pipeline
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

from .manhattan import estimate_manhattan, visualize_manhattan

# ── constants ─────────────────────────────────────────────────────────────────
VLM_API_URL      = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post
_N_ITER_DEFAULT  = 1
# Manhattan wireframe drawing
_EDGE_COLOUR     = (80, 200, 255)   # cyan-blue
_FLOOR_COLOUR    = (0,  255,   0)   # green  — floor anchor
_CEIL_COLOUR     = (0,  220, 255)   # cyan   — ceiling anchor
_ANCHOR_RADIUS   = 10
_EDGE_WIDTH      = 3


# ── small helpers ─────────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray, fallback=None) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n > 1e-9:
        return vec / n
    if fallback is None:
        raise ValueError("Cannot normalize near-zero vector")
    return _normalize(np.array(fallback, dtype=np.float64))


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_vlm_camera(walls_out_dir: str) -> dict:
    p = Path(walls_out_dir) / "camera.json"
    if not p.exists():
        raise FileNotFoundError(f"No camera.json in {walls_out_dir}")
    return json.loads(p.read_text())


def _load_vggt_intrinsics(vggt_out_dir: str, frame: int = 0) -> tuple[float, float, int, int]:
    """Return (fx, fy, W, H) in original pixel coordinates."""
    cam_json = _load_json(Path(vggt_out_dir) / "camera.json")
    cam = cam_json[frame] if isinstance(cam_json, list) else cam_json
    K = np.array(cam["intrinsic_3x3"], dtype=np.float64)
    H, W = cam["image_size_hw_orig"]
    return float(K[0, 0]), float(K[1, 1]), int(W), int(H)


def _load_vggt_image_path(vggt_out_dir: str, frame: int = 0) -> str | None:
    cam_json = _load_json(Path(vggt_out_dir) / "camera.json")
    cam = cam_json[frame] if isinstance(cam_json, list) else cam_json
    return cam.get("image")


def _find_wall_mesh(walls_out_dir: str) -> Path:
    root = Path(walls_out_dir)
    for p in [
        root / "openings" / "walls_with_glass.obj",
        root / "openings" / "walls_with_openings.obj",
        root / "walls.obj",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(f"No wall mesh found in {walls_out_dir}")


# ── anchor extraction ─────────────────────────────────────────────────────────

def _find_back_anchor(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the back vertical edge of the Manhattan box that corresponds to the
    deepest visible corner in the reference photo.

    Uses ``deepest_corner_idx`` (sampled from the actual depth map at each
    corner's projected pixel) rather than camera-Z of the 3-D corner positions.
    This correctly identifies the back corner that is *actually visible* in the
    reference image — not necessarily the corner whose 3-D position is farthest
    from the VGGT camera origin, which can be wrong when furniture occludes the
    true back corner.

    Returns (floor_corner_w, ceil_corner_w) in VGGT world space.
    """
    corners  = result["corners"]   # (8, 3)
    v_idx    = result["v_idx"]
    K, R, t  = result["K"], result["R"], result["t"]

    # deepest_corner_idx: the box corner whose depth-map sample is largest —
    # i.e. the back corner actually visible in the reference photo.
    deepest_idx = result.get("deepest_corner_idx")

    if deepest_idx is not None:
        # Identify the partner corner on the same vertical edge by flipping
        # the v_idx bit in the corner's bit-coded index.
        partner_idx = deepest_idx ^ (1 << v_idx)
        ci  = R @ corners[deepest_idx] + t
        cj  = R @ corners[partner_idx] + t
        W   = int(result["W"])
        # Both corners must be in front of the camera AND project within the
        # image width.  Off-screen corners indicate the anchor is not visible in
        # the reference photo; fall back to the camera-Z search in that case.
        visible = (
            ci[2] > 0 and cj[2] > 0
            and 0 <= float((K @ ci)[0] / ci[2]) < W
            and 0 <= float((K @ cj)[0] / cj[2]) < W
        )
        if visible:
            # OpenCV Y↓: larger cam_y = lower in image = floor
            if ci[1] > cj[1]:
                best_pair = (corners[deepest_idx].copy(), corners[partner_idx].copy())
            else:
                best_pair = (corners[partner_idx].copy(), corners[deepest_idx].copy())
            print(f"[align] Back anchor from deepest_corner_idx={deepest_idx} "
                  f"(partner={partner_idx})")
            floor_w, ceil_w = best_pair
            print(f"[align] Back anchor  floor (VGGT world): {np.round(floor_w, 3)}")
            print(f"[align] Back anchor  ceil  (VGGT world): {np.round(ceil_w,  3)}")
            return floor_w, ceil_w
        else:
            print(f"[align] deepest_corner_idx={deepest_idx} edge is off-screen "
                  f"— falling back to camera-Z search")

    # Fallback: deepest visible vertical edge by camera-Z (original behaviour)
    vertical_edges = [(i, j) for i, j, ax in result["edges"] if ax == v_idx]
    W = int(result["W"])
    best_zc   = -np.inf
    best_pair = None
    for i, j in vertical_edges:
        ci = R @ corners[i] + t
        cj = R @ corners[j] + t
        if ci[2] <= 0.0 or cj[2] <= 0.0:
            continue
        ux_i = float((K @ ci)[0] / ci[2])
        ux_j = float((K @ cj)[0] / cj[2])
        if not (0 <= ux_i < W and 0 <= ux_j < W):
            continue
        avg_zc = (float(ci[2]) + float(cj[2])) / 2.0
        if avg_zc > best_zc:
            best_zc = avg_zc
            if ci[1] > cj[1]:
                best_pair = (corners[i].copy(), corners[j].copy())
            else:
                best_pair = (corners[j].copy(), corners[i].copy())
    if best_pair is None:
        i, j = vertical_edges[0]
        ci = R @ corners[i] + t
        cj = R @ corners[j] + t
        if ci[1] > cj[1]:
            best_pair = (corners[i].copy(), corners[j].copy())
        else:
            best_pair = (corners[j].copy(), corners[i].copy())
        print("[align] WARNING: no visible vertical edge — using first as fallback anchor")

    floor_w, ceil_w = best_pair
    print(f"[align] Back anchor  floor (VGGT world): {np.round(floor_w, 3)}")
    print(f"[align] Back anchor  ceil  (VGGT world): {np.round(ceil_w,  3)}")
    return floor_w, ceil_w


def _detect_ceiling_visible(image_path: str) -> bool:
    """
    Ask the VLM whether the ceiling (or ceiling-wall junction) is visible
    in the reference photo.

    This determines whether the alignment loop should target both the floor
    AND ceiling anchors (True) or the floor anchor only (False).
    Falls back to True on any failure so we don't silently under-constrain.
    """
    img_b64 = _encode_image_b64(image_path)
    prompt = (
        "Look at this indoor room photo.\n"
        "Is the ceiling, or the junction where the ceiling meets a wall, "
        "clearly visible anywhere in the image?\n"
        "Answer with a single word: YES or NO."
    )
    payload = {
        "model": "qwen3",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        visible = raw.strip().upper().startswith("YES")
        print(f"[align] Ceiling visible in reference image (VLM): {visible}  "
              f"(raw: '{raw[:40]}')")
        return visible
    except Exception as e:
        print(f"[align] Ceiling visibility check failed: {e} — assuming visible.")
        return True


# ── Manhattan visualization helpers ───────────────────────────────────────────

def _far_edge_filter(result: dict) -> set[tuple[int, int]]:
    """
    Return the set of (min(i,j), max(i,j)) edge pairs to SHOW.

    Removed:
      - Ceiling horizontal edges  (both endpoints at ceiling level, axis ≠ v_idx)
      - Front-face edges          (both endpoints on the near side of the depth axis)

    What remains: the two far vertical edges, the back floor edge, and the two
    floor lines running depth-wise from the near side to the far corners.
    """
    axes     = result["axes"]
    v_idx    = result["v_idx"]
    planes   = result["planes"]
    R, t     = result["R"], result["t"]
    edges    = result["edges"]

    # Camera world position: R @ cam_w + t = 0  →  cam_w = -R.T @ t
    cam_w    = -(R.T @ t)

    # Identify depth axis: the horizontal axis most aligned with camera forward
    cam_fwd  = R.T @ np.array([0., 0., 1.])
    h_idxs   = [i for i in range(3) if i != v_idx]
    depth_ax = max(h_idxs, key=lambda a: float(abs(cam_fwd @ axes[a])))

    # Which side of the depth axis is nearest to the camera
    cam_d    = float(cam_w @ axes[depth_ax])
    d_lo, d_hi = planes[depth_ax]
    near_side  = 0 if abs(cam_d - d_lo) < abs(cam_d - d_hi) else 1

    # After orientation fix in estimate_manhattan, the vertical axis points "up"
    # (negative OpenCV-Y component), so the ceiling = high side = bit v_idx = 1.
    ceil_side = 1

    visible = set()
    for i, j, ax in edges:
        # Ceiling horizontal edges (both at ceiling level, not a vertical edge)
        if ax != v_idx:
            if ((i >> v_idx) & 1) == ceil_side and ((j >> v_idx) & 1) == ceil_side:
                continue

        # Front-face edges (both endpoints on the near side of the depth axis)
        if ((i >> depth_ax) & 1) == near_side and ((j >> depth_ax) & 1) == near_side:
            continue

        visible.add((min(i, j), max(i, j)))

    return visible


def _project_pt(pt_w: np.ndarray, K: np.ndarray,
                R: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """Project a world-space point → (u, v) using VGGT camera. None if behind."""
    c = R @ pt_w + t
    if c[2] <= 1e-4:
        return None
    h = K @ c
    return h[:2] / h[2]


def _project_render(pt_world: np.ndarray, cam: dict) -> np.ndarray | None:
    """
    Project a walls.obj world-space point through the render camera → (u, v) pixels.
    Uses architectural (two-point) perspective matching render_room.py: depth is
    computed along the horizontal forward (y=0), and vertical uses tilt correction.
    Returns None if the point is behind the camera.
    """
    pos   = np.array(cam["position_m"], dtype=np.float64)
    look  = np.array(cam["look_at_m"],  dtype=np.float64)
    W     = int(cam["width_px"])
    H     = int(cam["height_px"])
    hfov  = float(cam["hfov_deg"])

    fwd   = _normalize(look - pos)
    right = _normalize(np.cross(fwd, [0., 1., 0.]))

    # Architectural (two-point) perspective — matches render_room.py exactly.
    # Depth along horizontal forward; vertical corrected for camera tilt.
    fwd_h = np.array([fwd[0], 0.0, fwd[2]])
    fwd_h_n = float(np.linalg.norm(fwd_h))
    if fwd_h_n > 1e-9:
        fwd_h   /= fwd_h_n
        tilt_tan = fwd[1] / fwd_h_n
    else:
        fwd_h    = fwd.copy()
        tilt_tan = 0.0

    fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    cx, cy = W / 2.0, H / 2.0

    p_rel = np.asarray(pt_world, dtype=np.float64) - pos
    X_cam = float(np.dot(right, p_rel))
    Z_cam = float(np.dot(fwd_h, p_rel))  # architectural depth

    if Z_cam <= 1e-4:
        return None

    yc = float(p_rel[1]) - tilt_tan * Z_cam  # vertical with tilt correction

    u = fx * X_cam / Z_cam + cx
    v = cy - fx * yc / Z_cam
    return np.array([u, v])


def _analytical_camera_pin(
    cam: dict,
    floor_mesh_pt: np.ndarray,
    floor_target_px: np.ndarray,
    ceil_mesh_pt: np.ndarray | None = None,
    ceil_target_px: np.ndarray | None = None,
) -> dict:
    """
    Translate the camera so floor_mesh_pt projects exactly to floor_target_px.

    When both anchors are provided, a canonical HORIZONTAL camera is solved:
        Z_cam = fy * H_room / pixel_span   (horizontal pinhole, no tilt factor)
        cam_Y = -(floor_v - cy) * Z_cam / fy   (~1.3 m eye height, inside room)
    The current camera yaw is preserved; only pitch is flattened.  This ensures
    the camera stays inside the room regardless of the initial VLM camera tilt.

    Without ceiling: Z_cam is taken from the current camera (forward depth to
    floor_mesh_pt).  If the camera is too close the pin is skipped.
    """
    cam = dict(cam)
    pos  = np.array(cam["position_m"], dtype=np.float64)
    look = np.array(cam["look_at_m"],  dtype=np.float64)
    W    = int(cam["width_px"])
    H    = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])

    fwd   = _normalize(look - pos)
    right = _normalize(np.cross(fwd, [0., 1., 0.]))
    up    = np.cross(right, fwd)

    fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    # Use vfov for fy if available; otherwise assume square pixels (fx == fy).
    # The anamorphic formula (fx * H / W) is wrong for standard cameras.
    if "vfov_deg" in cam:
        fy = H / (2.0 * np.tan(np.radians(float(cam["vfov_deg"])) / 2.0))
    else:
        fy = fx
    cx, cy = W / 2.0, H / 2.0

    look_dist = max(float(np.linalg.norm(look - pos)), 0.5)

    min_z = 0.5
    world_up = np.array([0., 1., 0.])
    u_f, v_f = float(floor_target_px[0]), float(floor_target_px[1])

    if ceil_mesh_pt is not None and ceil_target_px is not None:
        # Canonical HORIZONTAL camera solve: ignore camera tilt entirely.
        # This guarantees cam_Y ≈ eye-height (inside room) regardless of initial pitch.
        #   Z_cam = fy * H_room / pixel_span   (horizontal pinhole formula)
        #   cam_Y = -(floor_v - cy) * Z_cam / fy
        H_room     = float(ceil_mesh_pt[1] - floor_mesh_pt[1])
        pixel_span = float(floor_target_px[1] - ceil_target_px[1])   # floor below → +
        if abs(H_room) > 0.01 and pixel_span > 5:
            Z_cam = fy * H_room / pixel_span
            print(f"[align] Pin (span, horiz): H_room={H_room:.3f}m  "
                  f"span={pixel_span:.1f}px  → Z_cam={Z_cam:.2f}m")
        else:
            Z_cam = float(np.dot(fwd, floor_mesh_pt - pos))

        if Z_cam < min_z:
            print(f"[align] Pin skipped: Z_cam={Z_cam:.2f}m too small")
            return cam

        # Flatten current forward direction to horizontal plane (preserve yaw from orbit)
        fwd_horiz   = _normalize(np.array([fwd[0], 0., fwd[2]]))
        right_horiz = _normalize(np.cross(fwd_horiz, world_up))

        X_cam_needed = (u_f - cx) * Z_cam / fx
        Y_cam_needed = -(v_f - cy) * Z_cam / fy   # Y-up: below-center → negative

        new_pos  = floor_mesh_pt - (X_cam_needed * right_horiz
                                    + Y_cam_needed * world_up
                                    + Z_cam * fwd_horiz)
        new_look = new_pos + look_dist * fwd_horiz

    else:
        # Architectural (two-point) perspective pin — matches render_room.py.
        # Depth along horizontal forward; vertical uses tilt correction so the
        # solved camera position projects the anchor to exactly (u_f, v_f).
        fwd_h_p = np.array([fwd[0], 0.0, fwd[2]])
        fwd_h_n_p = float(np.linalg.norm(fwd_h_p))
        if fwd_h_n_p > 1e-9:
            fwd_h_p  /= fwd_h_n_p
            tilt_tan_p = fwd[1] / fwd_h_n_p
        else:
            fwd_h_p    = fwd.copy()
            tilt_tan_p = 0.0

        Z_cam = float(np.dot(fwd_h_p, floor_mesh_pt - pos))
        if Z_cam < min_z:
            print(f"[align] Pin skipped: Z_cam={Z_cam:.2f}m too small")
            return cam

        X_cam_needed = (u_f - cx) * Z_cam / fx
        # Render: v = cy - fx * yc / Z_cam, yc = d_y - tilt_tan * Z_cam
        # Solve for d_y: d_y = -(v_f - cy) * Z_cam / fx + tilt_tan * Z_cam
        d_y_needed = -(v_f - cy) * Z_cam / fx + tilt_tan_p * Z_cam

        new_pos  = floor_mesh_pt - (X_cam_needed * right + d_y_needed * world_up
                                    + Z_cam * fwd_h_p)
        new_look = new_pos + look_dist * fwd

    print(f"[align] Pin: {np.round(pos,3)} → {np.round(new_pos,3)}  (Z_cam={Z_cam:.2f}m)")
    cam["position_m"] = new_pos.tolist()
    cam["look_at_m"]  = new_look.tolist()
    return cam


def _find_anchor_mesh_corner(
    mesh_path,
    cam: dict,
    target_px: np.ndarray,
    W_ref: int,
    H_ref: int,
) -> np.ndarray | None:
    """
    Find the back-wall floor corner in the mesh (min-Z, min-Y vertices) whose
    projection through the current render camera is closest to target_px.

    target_px is in reference-image pixel coordinates (W_ref × H_ref).
    Returns the best corner in walls.obj world coordinates, or None.
    """
    vertices = []
    try:
        with open(str(mesh_path)) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        return None
    if not vertices:
        return None

    verts = np.array(vertices)
    min_Y = float(verts[:, 1].min())
    min_Z = float(verts[:, 2].min())

    # Back-wall floor corners: low Y (floor) and low Z (back wall)
    mask = (verts[:, 1] < min_Y + 0.15) & (verts[:, 2] < min_Z + 0.30)
    candidates = verts[mask]
    if len(candidates) == 0:
        candidates = verts

    # Scale target_px to render resolution
    W_r = int(cam["width_px"])
    H_r = int(cam["height_px"])
    tpx = np.array([target_px[0] * W_r / W_ref, target_px[1] * H_r / H_ref])

    # ── Side-match pick (preferred when 2 back-wall candidates) ──────────────
    # VGGT's deepest_corner_idx selects a specific physical corner of the box;
    # the cyan target arms in the overlay radiate from THAT corner.  The mesh
    # candidates here are the floor-level back-wall corners (typically two for
    # rectangular meshes: back-LEFT at X=0 and back-RIGHT at X=mesh_W).  Picking
    # the one on the *same image-x side* as the VGGT-projected target pixel
    # ensures the rendered arms radiate from the corresponding physical corner,
    # so cyan and rendered arms align directionally instead of pointing in
    # opposite ways (which manifests as the "rotated around vertical" look the
    # closest-projection heuristic produced).
    cand_xs = [float(c[0]) for c in candidates]
    best_pt: np.ndarray | None = None
    if len(candidates) >= 2 and (max(cand_xs) - min(cand_xs)) > 0.01:
        image_cx = W_r / 2.0
        prefer_high_x = float(tpx[0]) > image_cx
        x_min, x_max = min(cand_xs), max(cand_xs)
        target_world_x = x_max if prefer_high_x else x_min
        best_dist_world = float("inf")
        for pt in candidates:
            d = abs(float(pt[0]) - target_world_x)
            if d < best_dist_world:
                best_dist_world = d
                best_pt = np.asarray(pt).copy()
        print(f"[align] Side-match pick: target_x_image={float(tpx[0]):.0f} "
              f"(cx={image_cx:.0f}) → {'high' if prefer_high_x else 'low'} X  "
              f"(candidates X={[round(x, 2) for x in cand_xs]})  "
              f"→ X={float(best_pt[0]):.2f}")
    else:
        # Single candidate or all-same-X: fall back to closest-projection
        best_dist = np.inf
        for pt in candidates:
            px = _project_render(pt, cam)
            if px is not None:
                dist = float(np.linalg.norm(px - tpx))
                if dist < best_dist:
                    best_dist = dist
                    best_pt = np.asarray(pt).copy()

    if best_pt is not None:
        px = _project_render(best_pt, cam)
        print(f"[align] Anchor mesh corner: {np.round(best_pt, 3)}  "
              f"renders at {np.round(px, 0) if px is not None else 'behind cam'}  "
              f"target px={np.round(tpx, 0)}")
    return best_pt


def _draw_box_and_anchors(
    draw: ImageDraw.ImageDraw,
    result: dict,
    floor_w: np.ndarray,
    ceil_w: np.ndarray,
    scale: float = 1.0,
    render_cam: dict | None = None,
    mesh_anchor_floor: np.ndarray | None = None,
    mesh_anchor_ceil:  np.ndarray | None = None,
    box_transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Draw the 12-edge Manhattan box wireframe + anchor markers on a PIL ImageDraw
    canvas.  Uses VGGT's own K/R/t for projection (original image space).

    Visual hierarchy:
      - Edges directly connected to an anchor corner → full brightness
      - All other edges → dimmed, so the VLM focuses on the anchor lines
      - Anchor markers → large filled circles with a white ring so they are
        unmistakably the alignment targets
    """
    corners = result["corners"]   # (8, 3)
    edges   = result["edges"]     # (i, j, ax)
    K, R, t = result["K"], result["R"], result["t"]

    # Find which corner indices correspond to the two anchors
    def _nearest_corner_idx(pt_w: np.ndarray) -> int:
        dists = np.linalg.norm(corners - pt_w, axis=1)
        return int(np.argmin(dists))

    floor_idx = _nearest_corner_idx(floor_w)
    ceil_idx  = _nearest_corner_idx(ceil_w)
    anchor_set = {floor_idx, ceil_idx}

    # When box_transform is supplied AND render_cam is supplied, the box
    # corners are first transformed into mesh world via box_transform = (M, t)
    # — i.e., corner_mesh = M @ corner_vggt + t — and then projected through
    # the render camera.  This makes the box corners coincide with the mesh
    # corners in 2D (assuming the transform was computed to do so).  This
    # supersedes FOV-aware mapping when the box is in mesh world.
    use_box_transform = (box_transform is not None and render_cam is not None)

    # When render_cam is supplied (without transform), build a FOV-aware
    # mapping from VGGT pixel space to render pixel space.  Simple resolution
    # scaling (rW/mW) is wrong when VGGT and render cameras have different
    # FOVs — same VGGT pixel column represents a different angular direction
    # in render space.  FOV-aware mapping uses the focal-length ratio so a
    # VGGT-projected ray lands at the render-pixel that corresponds to the
    # SAME angular direction.
    use_fov_map = (render_cam is not None
                   and "hfov_deg" in render_cam
                   and "width_px" in render_cam
                   and not use_box_transform)
    if use_fov_map:
        cx_v = float(K[0, 2]); cy_v = float(K[1, 2])
        fx_v = float(K[0, 0]); fy_v = float(K[1, 1])
        W_r  = int(render_cam["width_px"]);  H_r = int(render_cam["height_px"])
        hfov_r = float(render_cam["hfov_deg"])
        vfov_r = float(render_cam.get("vfov_deg", hfov_r))
        fx_r = W_r / (2.0 * np.tan(np.radians(hfov_r / 2.0)))
        fy_r = H_r / (2.0 * np.tan(np.radians(vfov_r / 2.0)))
        cx_r = W_r / 2.0;  cy_r = H_r / 2.0

        def _vggt_to_render(uv_vggt: np.ndarray) -> np.ndarray:
            u = (float(uv_vggt[0]) - cx_v) * (fx_r / fx_v) + cx_r
            v = (float(uv_vggt[1]) - cy_v) * (fy_r / fy_v) + cy_r
            return np.array([u, v])
    else:
        def _vggt_to_render(uv_vggt: np.ndarray) -> np.ndarray:
            return uv_vggt * scale

    # Project all 8 corners (VGGT) and map to render pixel space
    corners_px: list[np.ndarray | None] = []
    in_front:   list[bool]              = []

    if use_box_transform:
        # box_transform = (M, t):  corner_mesh = M @ corner_vggt + t.  Project
        # the transformed corner through the render camera.
        M_box, t_box = box_transform
        for c in corners:
            corner_mesh = M_box @ np.asarray(c, dtype=np.float64) + t_box
            px = _project_render(corner_mesh, render_cam)
            if px is None:
                corners_px.append(None)
                in_front.append(False)
            else:
                corners_px.append(np.asarray(px, dtype=np.float64))
                in_front.append(True)
    else:
        for c in corners:
            cc = R @ c + t
            in_front.append(bool(cc[2] > 0))
            if cc[2] > 1e-4:
                h = K @ cc
                uv_vggt = h[:2] / h[2]
                corners_px.append(_vggt_to_render(uv_vggt))
            else:
                corners_px.append(None)

    # Only show far-side edges (back wall + floor radiating lines, no ceiling/front)
    show_edges = _far_edge_filter(result)

    # Draw edges — anchor-adjacent edges bright, others dimmed
    _DIM_COLOUR = (40, 100, 130)   # muted version of _EDGE_COLOUR
    for i, j, _ in edges:
        if (min(i, j), max(i, j)) not in show_edges:
            continue
        if not (in_front[i] and in_front[j]):
            continue
        pi, pj = corners_px[i], corners_px[j]
        if pi is None or pj is None:
            continue
        is_anchor_edge = bool(anchor_set & {i, j})
        colour = _EDGE_COLOUR if is_anchor_edge else _DIM_COLOUR
        width  = _EDGE_WIDTH  if is_anchor_edge else max(1, _EDGE_WIDTH - 1)
        draw.line(
            [(float(pi[0]), float(pi[1])), (float(pj[0]), float(pj[1]))],
            fill=colour, width=width,
        )

    # Anchor markers — large filled circle + white ring so they stand out clearly
    # When render_cam + mesh anchors are supplied, draw markers at where the
    # mesh corners actually project through the render camera (so cyan/green
    # coincide with the rendered geometry's apex/ceiling by construction).
    # Otherwise fall back to VGGT's own projection of floor_w/ceil_w.
    use_mesh_markers = (
        render_cam is not None
        and mesh_anchor_floor is not None
        and mesh_anchor_ceil is not None
    )
    if use_mesh_markers:
        marker_specs: list[tuple[np.ndarray, tuple[int, int, int]]] = [
            (np.asarray(mesh_anchor_floor, dtype=np.float64), _FLOOR_COLOUR),
            (np.asarray(mesh_anchor_ceil,  dtype=np.float64), _CEIL_COLOUR),
        ]
    else:
        marker_specs = [(floor_w, _FLOOR_COLOUR), (ceil_w, _CEIL_COLOUR)]

    for pt_w, colour in marker_specs:
        if use_mesh_markers:
            # Project mesh-frame point through the render camera, NOT VGGT
            # (avoids the cyan/green-vs-rendered drift caused by VGGT's
            # tilted axes vs the mesh's axis-aligned convention).
            px = _project_render(pt_w, render_cam)  # type: ignore[arg-type]
            if px is None:
                continue
            x, y = float(px[0]), float(px[1])   # already in render-pixel space
        else:
            if use_box_transform:
                M_box, t_box = box_transform
                pt_mesh = M_box @ np.asarray(pt_w, dtype=np.float64) + t_box
                proj = _project_render(pt_mesh, render_cam)
                if proj is None:
                    continue
                x, y = float(proj[0]), float(proj[1])
            else:
                px = _project_pt(pt_w, K, R, t)
                if px is None:
                    continue
                mapped = _vggt_to_render(np.asarray(px, dtype=np.float64))
                x, y = float(mapped[0]), float(mapped[1])
        r = _ANCHOR_RADIUS
        # White ring (slightly larger) so the dot is visible on any background
        draw.ellipse([(x - r - 3, y - r - 3), (x + r + 3, y + r + 3)],
                     fill=(255, 255, 255), outline=(255, 255, 255))
        draw.ellipse([(x - r, y - r), (x + r, y + r)],
                     fill=colour, outline=(0, 0, 0), width=2)


def render_manhattan_reference(
    result: dict,
    floor_w: np.ndarray,
    ceil_w: np.ndarray,
    out_path: str,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """
    Draw the Manhattan box and anchor markers on a blank white background.
    Saved as manhattan_reference.png — fed to the VLM as the alignment target.
    """
    W, H = int(result["W"]), int(result["H"])
    img  = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)
    _draw_box_and_anchors(draw, result, floor_w, ceil_w, scale=1.0)
    img.save(out_path)
    print(f"[align] Manhattan reference → {out_path}")
    return out_path


def render_manhattan_overlay(
    render_path: str,
    result: dict,
    floor_w: np.ndarray,
    ceil_w: np.ndarray,
    out_path: str,
    render_cam: dict | None = None,
    mesh_anchor_floor: np.ndarray | None = None,
    mesh_anchor_ceil:  np.ndarray | None = None,
    box_transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    """
    Overlay the Manhattan box + anchor markers on a render image.

    By default, projects the VGGT box through VGGT's K/R/t — the cyan/green
    markers then sit where VGGT's reference projections place them.

    When ``render_cam`` and ``mesh_anchor_*`` are supplied, the cyan/green
    markers are instead drawn at the mesh corners' projections through the
    render camera.  In this mode the markers coincide with the rendered
    geometry's corners by construction (ie. they show the rendered apex/
    ceiling locations rather than VGGT's reference targets).
    """
    render = Image.open(render_path).convert("RGB")
    rW, _  = render.size
    mW     = int(result["W"])
    scale  = rW / mW   # uniform horizontal scale so projection maps correctly
    draw   = ImageDraw.Draw(render)
    _draw_box_and_anchors(
        draw, result, floor_w, ceil_w, scale=scale,
        render_cam=render_cam,
        mesh_anchor_floor=mesh_anchor_floor,
        mesh_anchor_ceil=mesh_anchor_ceil,
        box_transform=box_transform,
    )
    render.save(out_path)
    print(f"[align] Render + Manhattan overlay → {out_path}")
    return out_path


# ── VLM camera adjustment ─────────────────────────────────────────────────────

def _encode_image_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_vlm_for_camera_adjustment(
    manhattan_ref_path: str,
    render_overlay_path: str,
    ceil_visible: bool = True,
    floor_frac: float | None = None,
    ceil_frac: float | None = None,
) -> dict:
    """
    Ask the VLM to estimate:
      1. The pixel offset of the rendered back-wall floor corner from the GREEN
         anchor dot (used to translate the camera to pin the corner).
      2. The line-angle misalignment of the lines radiating from that corner
         (used to orbit the camera around the corner to align the directions).

    Returns dict with keys:
      corner_offset_px   : [du, dv]  — rendered corner displacement from GREEN dot
                           du > 0 → corner is to the RIGHT  of the dot
                           dv > 0 → corner is BELOW the dot
      line_yaw_delta_deg : orbit yaw  (positive = camera sweeps right around anchor)
      line_pitch_delta_deg: orbit pitch (positive = camera sweeps up around anchor)
    Falls back to zeros on any failure.
    """
    ref_b64    = _encode_image_b64(manhattan_ref_path)
    render_b64 = _encode_image_b64(render_overlay_path)

    # Numeric proportion hint so the VLM has a concrete vertical reference
    prop_hint = ""
    if floor_frac is not None:
        prop_hint = (
            f"\nAnchor vertical positions in the reference image "
            f"(0% = top, 100% = bottom):\n"
            f"  GREEN (floor corner): {floor_frac * 100:.1f}% from top\n"
        )
        if ceil_visible and ceil_frac is not None:
            prop_hint += (
                f"  CYAN (ceiling corner): {ceil_frac * 100:.1f}% from top\n"
            )

    anchor_scope = (
        "Both the GREEN (floor) and CYAN (ceiling) anchor corners."
        if ceil_visible else
        "The GREEN (floor) anchor corner only — ceiling is not in frame."
    )

    prompt = (
        "You are calibrating a 3D room render camera in two steps.\n\n"
        "Image 1 (REFERENCE): Manhattan wireframe on white background. "
        "GREEN = back-wall floor corner anchor. CYAN = ceiling anchor. "
        "Bright lines extend from the anchor corners — these are the alignment targets.\n\n"
        "Image 2 (RENDER + OVERLAY): Current render with the same wireframe overlaid.\n\n"
        f"{prop_hint}\n"
        f"Alignment scope: {anchor_scope}\n\n"
        "STEP 1 — Corner pixel offset\n"
        "Find the rendered back-wall floor corner in Image 2 "
        "(the innermost room floor corner visible in the render). "
        "Measure its pixel offset from the GREEN dot:\n"
        "  corner_offset_px = [du, dv]\n"
        "  du > 0 : rendered corner is to the RIGHT of the GREEN dot\n"
        "  du < 0 : rendered corner is to the LEFT\n"
        "  dv > 0 : rendered corner is BELOW the GREEN dot\n"
        "  dv < 0 : rendered corner is ABOVE\n"
        "Report [0, 0] if they already overlap.\n\n"
        "STEP 2 — Line angle misalignment (AFTER the corner is pinned)\n"
        "Assuming the corner is fixed at the GREEN dot, look at the angles of the "
        "bright lines radiating from it in Image 2 vs Image 1.\n"
        "  line_yaw_delta_deg   : >0 to sweep camera clockwise around the anchor "
        "(rotates the lines counter-clockwise in the image), <0 counter-clockwise\n"
        "  line_pitch_delta_deg : >0 to sweep camera upward around the anchor "
        "(tilts lines down in image), <0 to sweep downward\n"
        "Use 1–8° increments. Return 0 if lines already match.\n\n"
        "Respond with JSON only (no markdown fences):\n"
        '{"corner_offset_px": [<du>, <dv>], '
        '"line_yaw_delta_deg": <float>, '
        '"line_pitch_delta_deg": <float>, '
        '"reasoning": "<one sentence>"}'
    )

    payload = {
        "model": "qwen3",
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{render_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }

    try:
        resp = _vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            adj = json.loads(m.group())
            du, dv = adj.get("corner_offset_px", [0, 0])
            print(
                f"[align] VLM → corner_offset=({du:+.0f},{dv:+.0f})px  "
                f"orbit yaw={adj.get('line_yaw_delta_deg', 0):+.1f}°  "
                f"pitch={adj.get('line_pitch_delta_deg', 0):+.1f}°  "
                f"reason={str(adj.get('reasoning', ''))[:60]}"
            )
            return adj
        print(f"[align] VLM response could not be parsed as JSON.\nRaw: {raw[:300]}")
    except Exception as e:
        print(f"[align] VLM call failed: {e} — using zero adjustment.")
    return {"corner_offset_px": [0, 0],
            "line_yaw_delta_deg": 0.0,
            "line_pitch_delta_deg": 0.0}


def _geometric_orbit_correction(
    result: dict,
    floor_w: np.ndarray,
    anchor_mesh_pt: np.ndarray,
    cam: dict,
    target_px: np.ndarray,
    W_ref: int,
    H_ref: int,
    max_deg: float = 15.0,
    min_deg: float = 3.0,
) -> tuple[float, float]:
    """
    Compute (yaw_deg, pitch_deg) orbit corrections geometrically.

    Compares the angles of the two floor lines radiating from the anchor corner
    between:
      - Reference image  : VGGT projection of the Manhattan box floor edges
      - Current render   : render-camera projection of the mesh floor edge probes

    The two floor lines are classified as:
      'width' — back-wall floor edge (along horizontal axis ≠ depth axis)
      'depth' — side-wall floor edge (along depth axis, toward camera)

    Sign convention: yaw_deg > 0 means lines in the render need to rotate
    CLOCKWISE (matching the VLM convention in the prompt above).
    """
    corners = result["corners"]
    edges   = result["edges"]
    v_idx   = result["v_idx"]
    K, R, t = result["K"], result["R"], result["t"]

    # Determine depth axis (horizontal axis most aligned with camera forward)
    cam_fwd  = R.T @ np.array([0., 0., 1.])
    h_idxs   = [a for a in range(3) if a != v_idx]
    depth_ax = max(h_idxs, key=lambda a: abs(float(cam_fwd @ result["axes"][a])))

    # ── Reference angles from Manhattan ──────────────────────────────────────
    floor_idx     = int(np.argmin(np.linalg.norm(corners - floor_w, axis=1)))
    anchor_ref_px = _project_pt(floor_w, K, R, t)
    if anchor_ref_px is None:
        return 0.0, 0.0

    ref_angles: dict[str, float] = {}
    for i, j, ax in edges:
        if ax == v_idx:
            continue                         # skip vertical edge
        if i == floor_idx:
            nbr_idx = j
        elif j == floor_idx:
            nbr_idx = i
        else:
            continue
        nbr_px = _project_pt(corners[nbr_idx], K, R, t)
        if nbr_px is None:
            continue
        dx = float(nbr_px[0] - anchor_ref_px[0])
        dy = float(nbr_px[1] - anchor_ref_px[1])
        key = "depth" if ax == depth_ax else "width"
        ref_angles[key] = float(np.degrees(np.arctan2(dy, dx)))

    if not ref_angles:
        return 0.0, 0.0

    # ── Render angles from mesh probe points ─────────────────────────────────
    # Use the actual current projected position of the anchor (not the target pixel)
    # so angle measurements are consistent with where the anchor actually is in the render.
    anchor_render_px = _project_render(anchor_mesh_pt, cam)
    if anchor_render_px is None:
        return 0.0, 0.0

    render_angles: dict[str, float] = {}

    def _best_signed_angle(direction: np.ndarray, ref_angle: float) -> float | None:
        """
        Probe +direction and -direction from the anchor; return the signed angle
        (in degrees) of whichever probe is closer to ref_angle.
        This handles the case where the anchor is at the left or right back corner.
        """
        best_angle = None
        best_diff  = 180.1
        for sign in (1.0, -1.0):
            px = _project_render(anchor_mesh_pt + sign * direction, cam)
            if px is None:
                continue
            ddx = float(px[0] - anchor_render_px[0])
            ddy = float(px[1] - anchor_render_px[1])
            if abs(ddx) + abs(ddy) < 5:
                continue
            angle = float(np.degrees(np.arctan2(ddy, ddx)))
            diff  = abs(((angle - ref_angle) + 180.0) % 360.0 - 180.0)
            if diff < best_diff:
                best_diff  = diff
                best_angle = angle
        return best_angle

    # Width: back-wall floor edge (±X); pick sign matching ref
    if "width" in ref_angles:
        a = _best_signed_angle(np.array([1.0, 0.0, 0.0]), ref_angles["width"])
        if a is not None:
            render_angles["width"] = a

    # Depth: side-wall floor edge (±Z); pick sign matching ref
    if "depth" in ref_angles:
        a = _best_signed_angle(np.array([0.0, 0.0, 1.0]), ref_angles["depth"])
        if a is not None:
            render_angles["depth"] = a

    # ── Compute corrections ───────────────────────────────────────────────────
    # Width (back-wall floor edge) → yaw correction.
    # Depth (depth side floor edge) → pitch correction.
    # Same sign convention for both: diff > min_deg means render line angle
    # is `diff` degrees less than reference (in image-space arctan2, y-down),
    # which the orbit function interprets as needing CW image rotation
    # (positive yaw_deg / pitch_deg).
    yaw_deg   = 0.0
    pitch_deg = 0.0

    if "width" in ref_angles and "width" in render_angles:
        diff = float(ref_angles["width"] - render_angles["width"])
        diff = (diff + 180.0) % 360.0 - 180.0      # wrap to [-180, 180]
        if abs(diff) > min_deg:
            yaw_deg = float(np.clip(diff, -max_deg, max_deg))
        print(f"[align] Geo orbit width : ref={ref_angles['width']:.1f}°  "
              f"render={render_angles['width']:.1f}°  → yaw={yaw_deg:+.1f}°")

    if "depth" in ref_angles and "depth" in render_angles:
        diff = float(ref_angles["depth"] - render_angles["depth"])
        diff = (diff + 180.0) % 360.0 - 180.0
        if abs(diff) > min_deg:
            pitch_deg = float(np.clip(diff, -max_deg, max_deg))
        print(f"[align] Geo orbit depth : ref={ref_angles['depth']:.1f}°  "
              f"render={render_angles['depth']:.1f}°  → pitch={pitch_deg:+.1f}°")

    return yaw_deg, pitch_deg


# ── camera delta application ──────────────────────────────────────────────────

def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    K = np.array([
        [0.0,       -axis[2],  axis[1]],
        [axis[2],    0.0,     -axis[0]],
        [-axis[1],   axis[0],  0.0   ],
    ])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _apply_camera_corner_orbit(
    cam: dict,
    corner_du: float,
    corner_dv: float,
    line_yaw_deg: float,
    line_pitch_deg: float,
    initial_pos: np.ndarray | None = None,
    anchor_pt: np.ndarray | None = None,
    max_translate_m: float = 3.0,
    max_orbit_deg: float = 25.0,
) -> dict:
    """
    Two-phase camera adjustment to align a rendered corner with its reference.

    Phase 1 — Translation (pin the corner)
    ----------------------------------------
    Translate the camera perpendicular to the look direction so that the
    rendered corner shifts by (corner_du, corner_dv) pixels.  Both position
    and look_at move together (look direction preserved).

      move_right = corner_du  * look_dist / fx   (positive du → corner is right
                                                   of dot → camera slides right)
      move_up    = -corner_dv * look_dist / fy   (positive dv → corner is below
                                                   dot → camera slides up)

    Phase 2 — Orbit (align the lines)
    -----------------------------------
    Orbit the camera around the updated look_at (≈ the anchor world point) by
    (line_yaw_deg, line_pitch_deg).  The anchor stays at a fixed world position;
    the camera sweeps an arc around it like a compass arm, changing the apparent
    angles of the lines radiating from the corner.

      new_pos = anchor + R_orbit @ (pos - anchor)
      look_at remains = anchor

    Safety caps
    -----------
    - Total translation from initial position capped at max_translate_m.
    - Orbit angle per step capped at max_orbit_deg (prevents runaway iterations).
    """
    pos      = np.array(cam["position_m"], dtype=np.float64)
    look     = np.array(cam["look_at_m"],  dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0])
    fwd      = _normalize(look - pos)
    right    = _normalize(np.cross(fwd, world_up))
    up_cam   = np.cross(right, fwd)   # camera up (may differ from world_up if tilted)

    look_dist = max(float(np.linalg.norm(look - pos)), 0.5)

    # Focal length from hfov
    W    = int(cam["width_px"])
    H    = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    fy   = fx * H / W   # assume square pixels

    # ── Phase 1: translate to pin the corner ─────────────────────────────────
    if abs(corner_du) > 0.5 or abs(corner_dv) > 0.5:
        scale  = look_dist / fx
        delta  = right * (corner_du * scale) + up_cam * (-corner_dv * scale)
        pos   += delta
        look  += delta   # maintain look direction
        # Recompute frame after translation (look direction unchanged, but
        # we need updated right/up_cam for the orbit phase below)
        fwd    = _normalize(look - pos)
        right  = _normalize(np.cross(fwd, world_up))
        up_cam = np.cross(right, fwd)
        # Cap total translation drift
        if initial_pos is not None:
            drift = float(np.linalg.norm(pos - initial_pos))
            if drift > max_translate_m:
                pos   = initial_pos + (pos - initial_pos) * (max_translate_m / drift)
                look  = pos + look_dist * fwd
                print(f"[align] Translation capped: {drift:.2f}m → {max_translate_m}m")

    # ── Phase 2: orbit around anchor (look_at) ───────────────────────────────
    if abs(line_yaw_deg) > 0.5 or abs(line_pitch_deg) > 0.5:
        # Clamp per-step orbit angle
        line_yaw_deg   = max(-max_orbit_deg, min(max_orbit_deg, line_yaw_deg))
        line_pitch_deg = max(-max_orbit_deg, min(max_orbit_deg, line_pitch_deg))

        # Use the fixed anchor world point when available so the orbit pivot
        # doesn't drift as look_at accumulates translation corrections.
        anchor   = anchor_pt.copy() if anchor_pt is not None else look.copy()
        rel      = pos  - anchor
        look_rel = look - anchor   # look_at offset from anchor — rotated in sync

        # Positive line_yaw_delta_deg means lines in the render need to rotate
        # CLOCKWISE to match the reference.  Making lines rotate clockwise requires
        # sweeping the camera COUNTER-clockwise (negative rotation_y), hence the negation.
        R_yaw    = _rotation_y(np.radians(-line_yaw_deg))
        rel      = R_yaw @ rel
        look_rel = R_yaw @ look_rel   # keep look_at co-rotating with camera

        # Pitch: rotate around the right axis (after yaw)
        if abs(line_pitch_deg) > 0.5:
            fwd_after_yaw = _normalize(-rel)   # camera looks toward anchor
            right_after   = _normalize(np.cross(fwd_after_yaw, world_up))
            # Positive line_pitch_deg means floor lines need to tilt more steeply
            # downward, which requires the camera to move UP.  The rotation around
            # right_after with positive angle actually moves the camera DOWN, so
            # we negate to get the correct upward sweep.
            R_pitch  = _rotation_axis(right_after, np.radians(-line_pitch_deg))
            rel      = R_pitch @ rel
            look_rel = R_pitch @ look_rel   # keep look_at co-rotating with camera

        pos  = anchor + rel
        look = anchor + look_rel   # look_at rotates with camera — anchor stays near
                                   # its current pixel, so only a tiny pin is needed

        print(f"[align] Orbit yaw={line_yaw_deg:+.1f}°  pitch={line_pitch_deg:+.1f}°  "
              f"new pos={np.round(pos, 3)}")

    new_cam              = dict(cam)
    new_cam["position_m"] = pos.tolist()
    new_cam["look_at_m"]  = look.tolist()
    new_cam["up"]         = [0.0, 1.0, 0.0]
    return new_cam


# ── mesh FOV coverage extension ──────────────────────────────────────────────

def _extend_mesh_for_fov_coverage(
    mesh_path,
    cam: dict,
    render_path: str,
    bg_color: tuple[int, int, int] = (40, 40, 40),
    bg_tolerance: int = 20,
    margin_m: float = 0.5,
) -> bool:
    """
    Check the rendered image for void background near the left/right edges.

    Background at the sides means the side walls are too shallow in Z — the
    camera's frustum edge ray hits the side wall (X = cur_min/max_x) at a Z
    further than the wall extends.  Fix: push all front-face vertices (at
    Z = cur_max_z) out to Z where the frustum edge hits the side wall + margin.

    The back wall (Z = 0) and anchor corner are never touched, so camera
    alignment is fully preserved.  Returns True if the mesh was modified.
    """
    try:
        img = np.array(Image.open(str(render_path)).convert("RGB"), dtype=np.float32)
    except Exception as e:
        print(f"[align] FOV coverage check failed to load render: {e}")
        return False

    H_img, W_img = img.shape[:2]
    strip_w = max(1, W_img // 20)   # leftmost / rightmost 5 %
    bg = np.array(bg_color, dtype=np.float32)

    def _bg_fraction(strip: np.ndarray) -> float:
        return float((np.abs(strip - bg).max(axis=-1) < bg_tolerance).mean())

    left_bg  = _bg_fraction(img[:, :strip_w])  > 0.15
    right_bg = _bg_fraction(img[:, -strip_w:]) > 0.15

    if not (left_bg or right_bg):
        return False

    print(f"[align] Background visible at edges: left={left_bg}  right={right_bg}")

    pos   = np.array(cam["position_m"], dtype=np.float64)
    look  = np.array(cam["look_at_m"],  dtype=np.float64)
    hfov  = float(cam["hfov_deg"])

    fwd   = _normalize(look - pos)
    fwd_h = _normalize(np.array([fwd[0], 0., fwd[2]]))
    right = _normalize(np.cross(fwd_h, np.array([0., 1., 0.])))
    half  = float(np.tan(np.radians(hfov / 2.0)))

    # Read mesh vertices
    lines    = Path(str(mesh_path)).read_text().splitlines(keepends=True)
    verts_v  = [[float(p) for p in l.split()[1:4]] for l in lines if l.startswith("v ")]
    if not verts_v:
        return False
    verts_arr = np.array(verts_v)
    cur_min_x = float(verts_arr[:, 0].min())
    cur_max_x = float(verts_arr[:, 0].max())
    cur_max_z = float(verts_arr[:, 2].max())

    # For each side with background, shoot the outer frustum edge ray and find
    # the Z at which it hits the corresponding side wall (X = cur_min/max_x).
    # Extend all front-face vertices (Z ≈ cur_max_z) to that Z + margin.
    needed_max_z = cur_max_z

    def _z_hit_side_wall(x_wall: float, ray_dir: np.ndarray) -> float | None:
        """Z where ray (from camera pos, direction ray_dir) hits X = x_wall."""
        if abs(ray_dir[0]) < 1e-6:
            return None
        t = (x_wall - float(pos[0])) / float(ray_dir[0])
        if t <= 0:
            return None
        return float(pos[2]) + t * float(ray_dir[2])

    if left_bg:
        ray_left = _normalize(fwd_h - half * right)
        z_hit = _z_hit_side_wall(cur_min_x, ray_left)
        if z_hit is not None:
            needed_max_z = max(needed_max_z, z_hit + margin_m)
            print(f"[align] Left frustum hits side wall at Z={z_hit:.2f}m")

    if right_bg:
        ray_right = _normalize(fwd_h + half * right)
        z_hit = _z_hit_side_wall(cur_max_x, ray_right)
        if z_hit is not None:
            needed_max_z = max(needed_max_z, z_hit + margin_m)
            print(f"[align] Right frustum hits side wall at Z={z_hit:.2f}m")

    if needed_max_z <= cur_max_z + 0.01:
        return None

    # Move only front-face vertices (Z ≈ cur_max_z) to needed_max_z.
    # Back wall (Z = 0) and all interior vertices are untouched.
    z_tol = max((cur_max_z - 0) * 0.05, 0.05)
    new_lines = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            z = float(parts[3])
            new_z = needed_max_z if abs(z - cur_max_z) < z_tol else z
            new_lines.append(f"v {parts[1]} {parts[2]} {new_z:.6f}\n")
        else:
            new_lines.append(line)

    Path(str(mesh_path)).write_text("".join(new_lines))
    print(f"[align] Mesh Z extended: {cur_max_z:.2f} → {needed_max_z:.2f} m "
          f"(left={left_bg}  right={right_bg})")
    return needed_max_z


# ── multi-point camera optimization ──────────────────────────────────────────

def _collect_mesh_floor_corners(mesh_path: Path) -> np.ndarray | None:
    """Return Nx3 unique floor-level corners (Y near min) from walls.obj."""
    verts = []
    try:
        with open(mesh_path) as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split()
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
    except Exception:
        return None
    if not verts:
        return None
    arr = np.asarray(verts, dtype=np.float64)
    y_min = float(arr[:, 1].min())
    floor = arr[arr[:, 1] < y_min + 0.05]
    seen: dict[tuple[float, float], np.ndarray] = {}
    for v in floor:
        key = (round(float(v[0]), 3), round(float(v[2]), 3))
        if key not in seen:
            seen[key] = np.array([float(v[0]), y_min, float(v[2])])
    return np.array(list(seen.values()))


def _build_render_proj_fn():
    """Closure-free projection: given (pos, look, hfov, W, H, pt) → (u, v)
    or None if behind camera. Mirrors _project_render but takes scalars
    so least_squares can call it cheaply per-iteration."""
    def proj(pos: np.ndarray, look: np.ndarray, hfov: float,
             W: int, H: int, pt: np.ndarray) -> tuple[float, float] | None:
        fwd = look - pos
        n   = float(np.linalg.norm(fwd))
        if n < 1e-9:
            return None
        fwd /= n
        up_w = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up_w)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            return None
        right /= rn
        up = np.cross(right, fwd)
        fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        cx = W / 2.0
        cy = H / 2.0
        rel = pt - pos
        Xc = float(np.dot(right, rel))
        Yc = float(np.dot(up,    rel))
        Zc = float(np.dot(fwd,   rel))
        if Zc <= 1e-4:
            return None
        u = fx * Xc / Zc + cx
        v = cy - fx * Yc / Zc
        return (u, v)
    return proj


def _align_floor_first(
    cam: dict,
    result: dict,
    walls_out_dir: Path | str,
    mesh_path: Path | str | None = None,
    eye_height_m: float = 1.4,
    n_yaw_iter: int = 60,
    yaw_tol_deg: float = 0.1,
    _force_mesh_anchor_idx: int | None = None,
    image_path: "str | None" = None,
) -> dict:
    """
    Floor-first camera alignment.

    Pipeline:
      1. Pick the back-wall mesh anchor corner that matches the VGGT box
         deepest corner (closest-X-side heuristic).
      2. Seed cam at eye_height_m above the room centre, looking DOWN at
         that anchor (so cam direction is naturally tilted, not horizontal).
      3. Floor-only analytical pin: translate the cam (preserving direction)
         so the floor anchor projects exactly to its VGGT-target pixel.
         Because tilt is preserved, cam.y emerges around eye-height instead
         of collapsing to floor level — that's the whole point of going
         floor-first.
      4. Iterative yaw refinement: compare floor-line angles from the
         anchor between the VGGT projection and the current render via
         _geometric_orbit_correction; orbit the cam around the anchor;
         re-pin floor anchor.  Repeat until yaw delta < yaw_tol_deg.

    Ceiling-anchor pixel alignment is intentionally NOT enforced — under
    a horizontal-camera assumption, pinning both anchors collapses cam.y
    to the pixel-span ratio (which forces floor-level cam for many
    VGGT-estimated poses).  Skipping it lets cam.y stay at a sensible
    eye-height; the ceiling pixel is whatever the resulting pose gives.
    """
    walls_out_dir = Path(walls_out_dir)
    if mesh_path is None:
        try:
            mesh_path = _find_wall_mesh(str(walls_out_dir))
        except FileNotFoundError:
            print("[floor_first] no wall mesh — keeping input cam")
            return cam
    mesh_path = Path(mesh_path)

    mesh_floor = _collect_mesh_floor_corners(mesh_path)
    if mesh_floor is None or mesh_floor.shape[0] != 4:
        n = -1 if mesh_floor is None else int(mesh_floor.shape[0])
        print(f"[floor_first] expected 4 mesh floor corners, got {n} — keeping input cam")
        return cam

    bounds = _parse_mesh_bounds(str(mesh_path))

    # ── 1. Pick mesh anchor (back-wall corner matching VGGT deepest box corner) ─
    corners = np.asarray(result["corners"], dtype=np.float64)
    v_idx   = int(result["v_idx"])
    K_v, R_v, t_v = result["K"], result["R"], result["t"]

    deepest_full = int(result.get("deepest_corner_idx", 0))
    if ((deepest_full >> v_idx) & 1) != 0:
        deepest_full = deepest_full ^ (1 << v_idx)
    box_anchor_floor = corners[deepest_full]

    # Read VLM back_wall_side hint — overrides VGGT when deepest_corner_idx
    # puts the anchor on the wrong side (e.g. head-on symmetric views).
    _fa_path = walls_out_dir / "floorplan_analysis.json"
    _back_wall_side_vlm: "str | None" = None
    _vlm_corner_px: "float | None" = None
    _fa_data: "dict | None" = None
    if _fa_path.exists():
        try:
            _fa_data = json.loads(_fa_path.read_text())
            _back_wall_side_vlm = _fa_data.get("camera", {}).get("back_wall_side")
        except Exception:
            pass

    z_min = float(mesh_floor[:, 2].min())
    back_candidates = [i for i in range(4)
                       if abs(float(mesh_floor[i, 2]) - z_min) < 0.05]
    mesh_anchor_idx = back_candidates[0]
    if _force_mesh_anchor_idx is not None and _force_mesh_anchor_idx in back_candidates:
        # Retry call from the swap-anchor fallback below — use the explicitly
        # requested anchor instead of the heuristic.
        mesh_anchor_idx = _force_mesh_anchor_idx
        print(f"[floor_first] (retry) using forced mesh anchor idx={mesh_anchor_idx}")
    elif len(back_candidates) > 1:
        prefer_high_x: "bool | None" = None
        anchor_vggt_px = _project_pt(box_anchor_floor, K_v, R_v, t_v)
        if anchor_vggt_px is not None:
            W_seed    = int(cam.get("width_px",  int(result["W"])))
            hfov_seed = float(cam.get("hfov_deg", 70.0))
            fx_v_  = float(K_v[0, 0]); cx_v_ = float(K_v[0, 2])
            fx_seed = W_seed / (2.0 * np.tan(np.radians(hfov_seed / 2.0)))
            tu = (anchor_vggt_px[0] - cx_v_) * (fx_seed / fx_v_) + W_seed / 2.0
            prefer_high_x = bool(tu > (W_seed / 2.0))
        # VLM back_wall_side overrides VGGT when the anchor lands on the wrong side.
        # Also reads corner_px for floor_target_px x-override (applied after ~2495).
        if _back_wall_side_vlm in ("left", "right"):
            try:
                _vlm_corner_px = float(_fa_data.get("camera", {}).get("corner_px"))
            except (TypeError, ValueError):
                pass
            prefer_high_x_vlm = (_back_wall_side_vlm == "right")
            # The VLM corner_px pixel position is more reliable than its
            # back_wall_side text.
            # If corner_px is available and contradicts back_wall_side, trust
            # the pixel (corner left of image centre → low-x mesh anchor, right
            # of centre → high-x anchor).  This prevents a degenerate seed
            # where we project a high-x mesh corner to a low-x target pixel.
            if _vlm_corner_px is not None:
                _W_img = int(cam.get("width_px", int(result["W"])))
                _corner_prefer_high = (_vlm_corner_px > _W_img / 2.0)
                # corner_px within 10% of image centre is too ambiguous to
                # override back_wall_side — it's essentially centred and could
                # go either way due to rounding or a slightly off VLM estimate.
                _corner_near_centre = abs(_vlm_corner_px - _W_img / 2.0) < 0.10 * _W_img
                if _corner_near_centre:
                    print(f"[floor_first] corner_px={_vlm_corner_px:.0f} is near "
                          f"image centre — keeping back_wall_side={_back_wall_side_vlm!r}")
                elif _corner_prefer_high != prefer_high_x_vlm:
                    # corner_px contradicts back_wall_side. Also check whether
                    # it contradicts the VGGT anchor's own pixel side. Use a
                    # 60/40 split (looser than 2/3) so anchors near the 2/3
                    # boundary are still recognised as clearly off-centre.
                    _vggt_side_clear = (
                        prefer_high_x is not None
                        and (tu < 0.40 * _W_img or tu > 0.60 * _W_img)
                        and _corner_prefer_high != prefer_high_x
                    )
                    if _vggt_side_clear:
                        print(f"[floor_first] corner_px={_vlm_corner_px:.0f} contradicts "
                              f"back_wall_side={_back_wall_side_vlm!r} AND VGGT anchor "
                              f"(x={tu:.0f}) → trusting VGGT anchor side, "
                              f"prefer_high_x={prefer_high_x}")
                    else:
                        print(f"[floor_first] corner_px={_vlm_corner_px:.0f} contradicts "
                              f"back_wall_side={_back_wall_side_vlm!r} "
                              f"→ trusting corner_px, prefer_high_x={_corner_prefer_high}")
                        prefer_high_x_vlm = _corner_prefer_high
            if prefer_high_x is None or prefer_high_x_vlm != prefer_high_x:
                print(f"[floor_first] VLM corner hint overrides VGGT "
                      f"prefer_high_x={prefer_high_x} → {prefer_high_x_vlm}")
                prefer_high_x = prefer_high_x_vlm
                # Swap deepest_full to the sibling back-wall VGGT corner so that
                # box_anchor_ceil (ceiling target y) comes from the correct side.
                # back_wall_h = the horizontal axis that runs along the back wall.
                _h_idxs = [i for i in range(3) if i != v_idx]
                _cam_z_all = np.array(
                    [(R_v @ corners[i] + t_v)[2] for i in range(len(corners))]
                )
                _h_nbr_z = {h: float(_cam_z_all[deepest_full ^ (1 << h)])
                             for h in _h_idxs}
                _back_wall_h = max(_h_idxs, key=lambda h: _h_nbr_z[h])
                deepest_full     = deepest_full ^ (1 << _back_wall_h)
                box_anchor_floor = corners[deepest_full]
                print(f"[floor_first] swapped VGGT anchor to sibling "
                      f"idx={deepest_full} (back_wall_h={_back_wall_h})")
        if prefer_high_x is not None:
            cand_xs = [float(mesh_floor[i, 0]) for i in back_candidates]
            target_x = max(cand_xs) if prefer_high_x else min(cand_xs)
            for i in back_candidates:
                if abs(float(mesh_floor[i, 0]) - target_x) < 1e-3:
                    mesh_anchor_idx = i
                    break
    mesh_anchor_floor = mesh_floor[mesh_anchor_idx]
    print(f"[floor_first] mesh anchor: idx={mesh_anchor_idx} "
          f"xz=({mesh_anchor_floor[0]:.2f},{mesh_anchor_floor[2]:.2f})")

    # ── 2. VGGT FOV setup so render-pixel space == VGGT-pixel space ─────────
    fx_v = float(K_v[0, 0]); fy_v = float(K_v[1, 1])
    cx_v = float(K_v[0, 2]); cy_v = float(K_v[1, 2])
    W_v  = int(result["W"]); H_v = int(result["H"])
    hfov_vggt = float(np.degrees(2.0 * np.arctan(W_v / (2.0 * fx_v))))
    vfov_vggt = float(np.degrees(2.0 * np.arctan(H_v / (2.0 * fy_v))))

    # ── floor_target_px: where the mesh anchor should appear in the render ──────
    # Use the VLM camera's architectural projection of mesh_anchor_floor as the
    # x-anchor.  The VLM camera was placed specifically so the chosen mesh corner
    # appears at the Stage-1 corner pixel — making both render_vlm and
    # render_vggt anchored to the exact same 3D mesh corner at the same x pixel.
    # VGGT's own x estimate drifts when the Manhattan box is rotated or when
    # deepest_corner_idx is swapped, so we trust the VLM anchor.
    # The y-coordinate comes from VGGT (reliable for floor/ceiling height ratio).
    _floor_w_ref, _ceil_w_ref = _find_back_anchor(result)
    cc = R_v @ _floor_w_ref + t_v
    if cc[2] <= 1e-4:
        print("[floor_first] back anchor behind VGGT camera — keeping input cam")
        return cam
    floor_target_px = np.array([fx_v * cc[0] / cc[2] + cx_v,
                                fy_v * cc[1] / cc[2] + cy_v])
    cc_ceil = R_v @ _ceil_w_ref + t_v
    ceil_target_px = None
    if cc_ceil[2] > 1e-4:
        ceil_target_px = np.array([fx_v * cc_ceil[0] / cc_ceil[2] + cx_v,
                                   fy_v * cc_ceil[1] / cc_ceil[2] + cy_v])

    print(f"[floor_first] floor anchor target px (VGGT): "
          f"({floor_target_px[0]:.1f}, {floor_target_px[1]:.1f})")
    if ceil_target_px is not None:
        print(f"[floor_first] ceil  anchor target px (VGGT): "
              f"({ceil_target_px[0]:.1f}, {ceil_target_px[1]:.1f})")

    mesh_anchor_ceil = np.asarray(mesh_anchor_floor, dtype=np.float64).copy()
    mesh_anchor_ceil[1] += float(bounds["height_m"])

    # ── 3. Seed cam: eye-height above room centre, yaw from VLM camera ───────
    # Use the VLM camera's forward direction as the seed yaw when the anchor
    # is well in front of the VLM camera (architectural depth > 1m AND on-screen).
    # This reduces the initial orbit residual from 50°+ to a few degrees.
    # Fall back to looking straight at the anchor when the VLM direction is
    # degenerate for the current anchor (e.g. alternate anchor nearly perpendicular
    # to the VLM look direction).
    eye_h = float(mesh_anchor_floor[1]) + eye_height_m
    room_cx = (bounds["xmin"] + bounds["xmax"]) / 2.0
    room_cz = (bounds["zmin"] + bounds["zmax"]) / 2.0
    seed_pos = np.array([room_cx, eye_h, room_cz], dtype=np.float64)
    seed_look = np.asarray(mesh_anchor_floor, dtype=np.float64)  # default
    try:
        _vlm_p   = np.array(cam["position_m"], dtype=np.float64)
        _vlm_lk  = np.array(cam["look_at_m"],  dtype=np.float64)
        _vlm_fwd = _vlm_lk - _vlm_p
        _vlm_fwd_n = float(np.linalg.norm(_vlm_fwd))
        if _vlm_fwd_n > 1e-9:
            _vlm_fwd /= _vlm_fwd_n
            # Check architectural depth of mesh_anchor through VLM camera
            _fh = np.array([_vlm_fwd[0], 0., _vlm_fwd[2]])
            _fhn = float(np.linalg.norm(_fh))
            if _fhn > 1e-9:
                _fh /= _fhn
            _dv = np.asarray(mesh_anchor_floor, dtype=np.float64) - _vlm_p
            _zc = float(_dv @ _fh)
            _r  = np.cross(_vlm_fwd, np.array([0., 1., 0.]))
            _rn = float(np.linalg.norm(_r))
            if _rn > 1e-9:
                _r /= _rn
            _xc  = float(_dv @ _r)
            _W_c = int(cam.get("width_px", W_v))
            _fx_c = _W_c / (2.0 * np.tan(np.radians(float(cam.get("hfov_deg", 70.0)) / 2.0)))
            _u   = _W_c / 2.0 + _fx_c * _xc / max(_zc, 1e-4)
            # Only use VLM direction when anchor has reasonable depth AND appears
            # near the centre of the VLM image (within central 60%).  If the
            # anchor is in the outer 20% on either side, the VLM camera is
            # facing mostly away from the anchor — using that direction as seed
            # points the camera outside the room.
            _u_margin = 0.20 * _W_c
            if _zc > 1.0 and _u_margin <= _u <= _W_c - _u_margin:
                _look_dist_seed = max(_vlm_fwd_n, 0.5)
                seed_look = seed_pos + _look_dist_seed * _vlm_fwd
                print(f"[floor_first] seed yaw from VLM direction "
                      f"(anchor Z_cam={_zc:.2f}m, u={_u:.0f}px)")
            else:
                print(f"[floor_first] seed yaw: straight at anchor "
                      f"(VLM Z_cam={_zc:.2f}m or u={_u:.0f} degenerate)")
    except Exception:
        pass

    cam_seed = dict(cam)
    cam_seed["position_m"] = seed_pos.tolist()
    cam_seed["look_at_m"]  = seed_look.tolist()
    cam_seed["hfov_deg"]   = hfov_vggt
    cam_seed["vfov_deg"]   = vfov_vggt
    cam_seed["width_px"]   = W_v
    cam_seed["height_px"]  = H_v
    cam_seed["up"]         = [0.0, 1.0, 0.0]
    print(f"[floor_first] seed: pos={np.round(seed_pos, 3).tolist()}  "
          f"look_at={np.round(seed_look, 3).tolist()}  (down-tilted at floor anchor)")

    # ── 4. Floor-only analytical pin (preserves cam direction → preserves tilt) ─
    cam_pinned = _analytical_camera_pin(
        cam_seed,
        floor_mesh_pt   = mesh_anchor_floor,
        floor_target_px = floor_target_px,
    )
    print(f"[floor_first] after floor pin: "
          f"pos={np.round(cam_pinned['position_m'], 3).tolist()}  "
          f"look_at={np.round(cam_pinned['look_at_m'], 3).tolist()}")

    # ── 5+6. Outer loop alternating yaw refinement + zoom-back ──────────────
    # Phase-1 (yaw): orbit camera around the floor anchor to align floor-wall
    # line angles with the reference. Floor anchor pin preserved by orbit.
    # Phase-2 (zoom): slide camera along its own look direction (anchored by
    # re-pinning the floor corner each step) so the ceiling corner moves to
    # its VGGT-target pixel.  Sliding along ±fwd changes Z_cam to every
    # world point by the same Δd but leaves X_cam, Y_cam unchanged, so it's
    # the cleanest one-DOF knob for moving the ceiling pixel without
    # rotating the camera.
    #
    # The two phases interact: yaw orbit changes which side of the room
    # the cam looks at (affects ceiling pixel); zoom-back changes
    # perspective foreshortening (affects floor-line angles).  Wrap them
    # in an outer loop so each phase's drift is corrected by the next
    # pass — typically converges in 2–3 outer iterations.
    #
    # Sign convention reconciliation: _geometric_orbit_correction returns
    # diff = (ref_angle - render_angle) in image-space arctan2 (y-down). It
    # claims "+ = render needs CW rotation", and _apply_camera_corner_orbit
    # negates that internally to orbit the camera CCW. Empirically, for
    # eye-height tilted-down cameras this composition produces an image
    # rotation in the OPPOSITE direction of `diff` — so we negate on the
    # way in.
    floor_w_v, _    = _find_back_anchor(result)
    initial_pos     = np.array(cam_pinned["position_m"], dtype=np.float64)
    # Iteration counts are deliberately high — we want the loops to keep
    # going until convergence or stall, not stop at an arbitrary count.
    # Stall detection (residual not shrinking) prevents infinite loops.
    n_outer_iter    = 30
    ceil_tol_px     = 1.0
    n_zoom_iter     = 30
    max_zoom_step_m = 1.5
    # Hard physical bounds on camera height. The orbit + zoom math has no
    # built-in awareness of floor.y or ceiling.y, so without these guards
    # a pathological scene can drive the camera underground (pointing up
    # at the ceiling) or above the ceiling (pointing down at the floor)
    # while still satisfying the floor-anchor pin in pixel space.
    # 5 cm of clearance on each side is just enough to avoid degenerate
    # math (cam exactly at floor / ceiling) while still allowing the
    # algorithm to put the camera anywhere physically inside the room.
    floor_y_world    = float(bounds["ymin"])
    ceil_y_world     = float(bounds["ymax"])
    min_cam_y_world  = floor_y_world + 0.05
    max_cam_y_world  = ceil_y_world  - 0.05
    # Z bounds: camera must stay in front of the back wall (zmin) and not fly
    # out the front.  Allow 1× room-depth of slack on the far side so cameras
    # looking at wide-angle rooms still converge; the back-wall constraint
    # (z > zmin) is hard — going behind it renders void and flips all angles.
    back_z_world  = float(bounds["zmin"])
    front_z_world = float(bounds["zmax"])
    room_depth_z  = max(front_z_world - back_z_world, 0.5)

    # Captured at end of phase-1 outer-iter-1 (see snapshot below); written
    # into the returned cam dict so the caller can render a "floor-aligned"
    # preview from the *winning* anchor's intermediate state, not whichever
    # anchor we tried first.
    floor_only_position_m: list[float] | None = None
    floor_only_look_at_m:  list[float] | None = None

    for outer_it in range(n_outer_iter):
        # ── Phase 1 (yaw + pitch — back-wall and depth floor edges) ─────────
        # The orbit function's pitch and yaw sign conventions are not
        # reliable across scenes (depends on which corner is the anchor,
        # which side of the room the cam sits on, etc.). Instead of
        # hard-coding signs, try all 4 combinations of (yaw_sign, pitch_sign)
        # each iteration and accept the one that actually reduces the
        # residual.  Cheap (4 trial orbits, no rendering) and robust.
        # Pass max_deg=180° so the returned deltas are the TRUE residuals
        # (not clipped to 15°), which lets stall detection work correctly.
        def _orbit_residual(c: dict) -> tuple[float, float]:
            # Pass min_deg=0 so the function reports the raw (unclipped,
            # non-dead-zoned) delta. We do convergence and 4-way picking
            # against tol externally. max_deg=180 disables the ±15° clip
            # so we see the full residual magnitude for stall detection.
            return _geometric_orbit_correction(
                result, floor_w_v, mesh_anchor_floor, c,
                target_px=floor_target_px, W_ref=W_v, H_ref=H_v,
                max_deg=180.0, min_deg=0.0,
            )

        yaw_did_anything = False
        for it in range(n_yaw_iter):
            yaw_d, pitch_d = _orbit_residual(cam_pinned)
            abs_delta = max(abs(yaw_d), abs(pitch_d))
            if abs_delta < yaw_tol_deg:
                print(f"[floor_first] outer {outer_it+1} floor-edges converged "
                      f"at inner {it+1} (yaw={yaw_d:+.2f}° pitch={pitch_d:+.2f}°)")
                break

            # 4-way sign trial. Reject any candidate that places the camera
            # outside the physical room (below floor or above ceiling).
            candidates: list[tuple[float, dict, int, int, float, float]] = []
            rejected_oob = 0
            # Adaptive per-step cap: scale with current residual so big initial
            # residuals (40°+) converge in a few steps without overshooting
            # near zero.  Range: 2° (close to converged) to 10° (far from it).
            step_cap = float(np.clip(0.3 * abs_delta, 2.0, 10.0))
            for sy in (1, -1):
                for sp in (1, -1):
                    ay = float(np.clip(sy * yaw_d,   -step_cap, step_cap))
                    ap = float(np.clip(sp * pitch_d, -step_cap, step_cap))
                    tc = _apply_camera_corner_orbit(
                        cam_pinned,
                        corner_du=0.0, corner_dv=0.0,
                        line_yaw_deg=ay,
                        line_pitch_deg=ap,
                        initial_pos=initial_pos,
                        anchor_pt=mesh_anchor_floor,
                        max_translate_m=1.0,
                        max_orbit_deg=step_cap,
                    )
                    tc = _analytical_camera_pin(
                        tc,
                        floor_mesh_pt   = mesh_anchor_floor,
                        floor_target_px = floor_target_px,
                    )
                    cam_y = float(tc["position_m"][1])
                    cam_z = float(tc["position_m"][2])
                    cam_x = float(tc["position_m"][0])
                    if cam_y < min_cam_y_world or cam_y > max_cam_y_world:
                        rejected_oob += 1
                        continue
                    if cam_z < back_z_world or cam_z > front_z_world + room_depth_z:
                        rejected_oob += 1
                        continue
                    if cam_x < bounds["xmin"] - 0.1 or cam_x > bounds["xmax"] + 0.1:
                        rejected_oob += 1
                        continue
                    ty, tp = _orbit_residual(tc)
                    tr = max(abs(ty), abs(tp))
                    candidates.append((tr, tc, sy, sp, ay, ap))

            if not candidates:
                print(f"[floor_first] outer {outer_it+1} floor-edges stopped "
                      f"at inner {it+1}: all 4 sign combos place cam outside "
                      f"room (y∉[{min_cam_y_world:.2f}, {max_cam_y_world:.2f}]m)")
                break

            candidates.sort(key=lambda x: x[0])
            best_r, best_cam, best_sy, best_sp, best_ay, best_ap = candidates[0]

            stall_eps = max(0.02, 0.01 * abs_delta)
            if best_r > abs_delta - stall_eps:
                print(f"[floor_first] outer {outer_it+1} floor-edges stalled "
                      f"at inner {it+1} (yaw={yaw_d:+.2f}° pitch={pitch_d:+.2f}° "
                      f"best 4-way trial residual={best_r:.2f}° "
                      f"prev={abs_delta:.2f}° "
                      f"rejected_oob={rejected_oob})")
                break

            cam_pinned    = best_cam
            yaw_did_anything = True
            print(f"[floor_first] outer {outer_it+1} floor-edges inner {it+1}: "
                  f"true yaw={yaw_d:+.2f}° pitch={pitch_d:+.2f}° "
                  f"signs=(y{best_sy:+d}, p{best_sp:+d}) "
                  f"applied yaw={best_ay:+.2f}° pitch={best_ap:+.2f}°  "
                  f"residual {abs_delta:.2f}°→{best_r:.2f}°  "
                  f"pos={np.round(cam_pinned['position_m'], 3).tolist()}")

        # Snapshot the "floor-only" intermediate the first time phase-1
        # finishes — floor anchor + floor-wall lines are aligned, ceiling
        # is not yet touched. Saved to the returned cam dict so the caller
        # can render a comparison preview AFTER the swap-anchor retry has
        # decided which attempt's intermediate is the winning one.
        if outer_it == 0:
            floor_only_position_m = list(cam_pinned["position_m"])
            floor_only_look_at_m  = list(cam_pinned["look_at_m"])

        # ── Phase 2 (zoom) ──────────────────────────────────────────────────
        zoom_did_anything = False
        # Skip ceiling zoom when the ceiling anchor is within 3% of the image
        # edge (top or bottom).  At such extremes a 1px camera move causes
        # enormous apparent pixel shifts → the zoom oscillates and drives the
        # camera outside the room trying to satisfy an inherently unstable
        # constraint.  Scipy polish handles residual ceil_dv in those cases.
        _ceil_edge_margin = 0.03 * H_v
        _ceil_in_bounds = (
            ceil_target_px is not None
            and float(ceil_target_px[1]) > _ceil_edge_margin
            and float(ceil_target_px[1]) < H_v - _ceil_edge_margin
        )
        if _ceil_in_bounds:
            ceil_target_v = float(ceil_target_px[1])
            last_abs_dv   = float("inf")
            for it in range(n_zoom_iter):
                pos_v   = np.asarray(cam_pinned["position_m"], dtype=np.float64)
                look_v  = np.asarray(cam_pinned["look_at_m"],  dtype=np.float64)
                fwd_v   = look_v - pos_v
                fn      = float(np.linalg.norm(fwd_v))
                if fn < 1e-6:
                    break
                fwd_v   = fwd_v / fn
                right_v = np.cross(fwd_v, np.array([0.0, 1.0, 0.0]))
                rn      = float(np.linalg.norm(right_v))
                if rn < 1e-9:
                    break
                right_v = right_v / rn
                up_v    = np.cross(right_v, fwd_v)

                rel_ceil = mesh_anchor_ceil - pos_v
                Y_ceil   = float(np.dot(up_v,  rel_ceil))
                Z_ceil   = float(np.dot(fwd_v, rel_ceil))
                if Z_ceil <= 0.1 or abs(Y_ceil) < 1e-3:
                    print(f"[floor_first] outer {outer_it+1} zoom skipped: "
                          f"degenerate ceiling geometry "
                          f"(Y={Y_ceil:.3f} Z={Z_ceil:.3f})")
                    break

                fx_r = W_v / (2.0 * np.tan(np.radians(hfov_vggt / 2.0)))
                cy_r = H_v / 2.0
                ceil_v_curr = cy_r - fx_r * Y_ceil / Z_ceil
                delta_v = ceil_v_curr - ceil_target_v
                if abs(delta_v) < ceil_tol_px:
                    print(f"[floor_first] outer {outer_it+1} ceiling converged "
                          f"at zoom inner {it+1} (delta_v={delta_v:+.1f}px)")
                    break
                # Stall detection: bail only when |delta_v| stops shrinking
                # by at least 0.05 px (or 1% of previous, whichever is larger).
                stall_eps_px = max(0.05, 0.01 * last_abs_dv)
                if abs(delta_v) > last_abs_dv - stall_eps_px:
                    print(f"[floor_first] outer {outer_it+1} ceiling stalled "
                          f"at zoom inner {it+1} (delta_v={delta_v:+.1f}px "
                          f"prev={last_abs_dv:.1f}px)")
                    break
                last_abs_dv = abs(delta_v)

                denom = cy_r - ceil_target_v
                if abs(denom) < 1e-3:
                    break
                Z_target = fx_r * Y_ceil / denom
                delta_d  = float(np.clip(Z_target - Z_ceil,
                                         -max_zoom_step_m, max_zoom_step_m))

                new_pos  = pos_v  - delta_d * fwd_v
                new_look = look_v - delta_d * fwd_v
                trial_cam = dict(cam_pinned)
                trial_cam["position_m"] = new_pos.tolist()
                trial_cam["look_at_m"]  = new_look.tolist()
                trial_cam = _analytical_camera_pin(
                    trial_cam,
                    floor_mesh_pt   = mesh_anchor_floor,
                    floor_target_px = floor_target_px,
                )
                cam_y_trial = float(trial_cam["position_m"][1])
                cam_z_trial = float(trial_cam["position_m"][2])
                cam_x_trial = float(trial_cam["position_m"][0])
                if cam_y_trial < min_cam_y_world or cam_y_trial > max_cam_y_world:
                    print(f"[floor_first] outer {outer_it+1} zoom stopped at "
                          f"inner {it+1}: zoom step would place cam at "
                          f"y={cam_y_trial:.2f}m, outside room bounds "
                          f"[{min_cam_y_world:.2f}, {max_cam_y_world:.2f}]m")
                    break
                if cam_z_trial < back_z_world or cam_z_trial > front_z_world + room_depth_z:
                    print(f"[floor_first] outer {outer_it+1} zoom stopped at "
                          f"inner {it+1}: zoom step would place cam at "
                          f"z={cam_z_trial:.2f}m, outside z bounds "
                          f"[{back_z_world:.2f}, {front_z_world + room_depth_z:.2f}]m")
                    break
                if cam_x_trial < bounds["xmin"] - 0.5 or cam_x_trial > bounds["xmax"] + 0.5:
                    print(f"[floor_first] outer {outer_it+1} zoom stopped at "
                          f"inner {it+1}: zoom step would place cam at "
                          f"x={cam_x_trial:.2f}m, outside x bounds "
                          f"[{bounds['xmin'] - 0.5:.2f}, {bounds['xmax'] + 0.5:.2f}]m")
                    break
                cam_pinned = trial_cam
                zoom_did_anything = True
                print(f"[floor_first] outer {outer_it+1} zoom inner {it+1}: "
                      f"delta_v={delta_v:+.1f}px  step={delta_d:+.2f}m  "
                      f"pos={np.round(cam_pinned['position_m'], 3).tolist()}")

        # Re-pin floor anchor after zoom: sliding along fwd changes Z_cam,
        # drifting the anchor's projected x. Re-pinning corrects this so the
        # floor pixel stays at floor_target_px regardless of zoom distance.
        if zoom_did_anything:
            cam_pinned = _analytical_camera_pin(
                cam_pinned,
                floor_mesh_pt   = mesh_anchor_floor,
                floor_target_px = floor_target_px,
            )

        if not yaw_did_anything and not zoom_did_anything:
            print(f"[floor_first] outer {outer_it+1}: neither phase made "
                  f"progress — exiting outer loop")
            break

    # ── Joint scipy polish on ALL 4 image-space targets ──────────────────────
    # After the iterative loop converges (or stalls), run a global L-BFGS-B
    # over all 6 DOFs (cam pos + look_at) with cam.y bounded in
    # [min_cam_y, max_cam_y]. Loss = floor anchor pixel² + ceiling anchor
    # pixel² + width angle deg² + depth angle deg². If a feasible global
    # minimum exists, scipy finds it; if not, it reports the best
    # constrained compromise.
    try:
        from scipy.optimize import minimize as _scipy_minimize
    except Exception as _scipy_e:
        print(f"[floor_first] scipy unavailable ({_scipy_e}) — skipping polish")
    else:
        proj_fn = _build_render_proj_fn()
        ceil_target_arr = (np.asarray(ceil_target_px, dtype=np.float64)
                           if ceil_target_px is not None else None)
        floor_target_arr = np.asarray(floor_target_px, dtype=np.float64)

        def _polish_loss(x: np.ndarray) -> float:
            pos = np.array([x[0], x[1], x[2]])
            look = np.array([x[3], x[4], x[5]])
            BEHIND = 1e8
            tot = 0.0

            anc = proj_fn(pos, look, hfov_vggt, W_v, H_v, mesh_anchor_floor)
            if anc is None:
                return BEHIND
            tot += float((anc[0] - floor_target_arr[0]) ** 2
                         + (anc[1] - floor_target_arr[1]) ** 2)

            if ceil_target_arr is not None:
                ce = proj_fn(pos, look, hfov_vggt, W_v, H_v, mesh_anchor_ceil)
                if ce is None:
                    return BEHIND
                tot += float((ce[0] - ceil_target_arr[0]) ** 2
                             + (ce[1] - ceil_target_arr[1]) ** 2)

            cam_trial = dict(cam_pinned)
            cam_trial["position_m"] = pos.tolist()
            cam_trial["look_at_m"]  = look.tolist()
            yaw_r, pitch_r = _orbit_residual(cam_trial)
            # Pixel² and degree² are both O(100) for typical residuals, so
            # equal-weighted is reasonable. 100× on angle gives slight
            # priority to closing the angle gap once pixels are close.
            tot += 100.0 * (yaw_r * yaw_r + pitch_r * pitch_r)
            return tot

        pos0_polish  = np.asarray(cam_pinned["position_m"], dtype=np.float64)
        look0_polish = np.asarray(cam_pinned["look_at_m"],  dtype=np.float64)
        x0_polish    = np.concatenate([pos0_polish, look0_polish])
        bounds_polish = [
            (None, None),                          # pos.x
            (min_cam_y_world, max_cam_y_world),    # pos.y — physical bound
            (None, None),                          # pos.z
            (None, None),                          # look.x
            (None, None),                          # look.y
            (None, None),                          # look.z
        ]
        initial_loss = _polish_loss(x0_polish)
        try:
            polish_res = _scipy_minimize(
                _polish_loss, x0_polish,
                method="L-BFGS-B", bounds=bounds_polish,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
            )
            final_polish_loss = float(polish_res.fun)
            if final_polish_loss < initial_loss * 0.99:
                cam_pinned = dict(cam_pinned)
                cam_pinned["position_m"] = polish_res.x[:3].tolist()
                cam_pinned["look_at_m"]  = polish_res.x[3:].tolist()
                # Floor anchor may have drifted by a fraction of a pixel
                # under joint optimization; re-pin to nail it exactly.
                cam_pinned = _analytical_camera_pin(
                    cam_pinned,
                    floor_mesh_pt   = mesh_anchor_floor,
                    floor_target_px = floor_target_px,
                )
                print(f"[floor_first] scipy polish: loss "
                      f"{initial_loss:.0f}→{final_polish_loss:.0f}  "
                      f"pos={np.round(cam_pinned['position_m'], 3).tolist()}")
            else:
                print(f"[floor_first] scipy polish: no improvement "
                      f"({initial_loss:.0f}→{final_polish_loss:.0f})")
        except Exception as _po_e:
            print(f"[floor_first] scipy polish raised ({_po_e}) — "
                  f"keeping iterative result")

    # ── Ceiling-height adjustment pass ──────────────────────────────────────
    # After yaw+zoom+scipy: if the ceiling is still above/invisible (projected
    # pixel < target, i.e. ceiling appears too high in image), the VLM-estimated
    # ceiling height is likely too large. Solve analytically for the height that
    # brings the ceiling to the target with the current camera, then re-run just
    # the zoom phase (floor anchor stays pinned) so the camera settles at the
    # correct depth for the adjusted height. Up to 5 adjustment attempts.
    if ceil_target_px is not None:
        _ceil_h_adj_tol = 2.0   # px: only adjust if further than 2px from target
        for _ch_it in range(5):
            _p  = np.asarray(cam_pinned["position_m"], dtype=np.float64)
            _l  = np.asarray(cam_pinned["look_at_m"],  dtype=np.float64)
            _fv = _normalize(_l - _p)
            if _fv is None:
                break
            _rv = _normalize(np.cross(_fv, np.array([0., 1., 0.])))
            if _rv is None:
                break
            _uv = np.cross(_rv, _fv)
            _rc = mesh_anchor_ceil - _p
            _Yc = float(np.dot(_uv, _rc))
            _Zc = float(np.dot(_fv, _rc))
            if _Zc <= 1e-3 or abs(_Yc) < 1e-3:
                break
            _fx = W_v / (2. * np.tan(np.radians(hfov_vggt / 2.)))
            _fy = H_v / (2. * np.tan(np.radians(vfov_vggt / 2.)))
            _cy = H_v / 2.
            _cv = _cy - _fy * _Yc / _Zc
            _dv = _cv - float(ceil_target_px[1])
            if abs(_dv) < _ceil_h_adj_tol:
                break   # already close enough
            # Analytical solve: find height h such that
            #   (anchor_x, h, anchor_z) → ceil_target_px[1]
            # with the current camera.
            _px, _py, _pz = _p
            _ax = float(mesh_anchor_floor[0])
            _bz = float(mesh_anchor_floor[2])
            _ac = (_ax - _px)*_fv[0] - _py*_fv[1] + (_bz - _pz)*_fv[2]
            _bc = _fv[1]
            _ec = (_ax - _px)*_uv[0] - _py*_uv[1] + (_bz - _pz)*_uv[2]
            _fc = _uv[1]
            _vt = float(ceil_target_px[1])
            _dn = _fy * _fc - (_cy - _vt) * _bc
            if abs(_dn) < 1e-9:
                break
            _hn = ((_cy - _vt) * _ac - _fy * _ec) / _dn
            if not (1.5 <= _hn <= 4.5):
                print(f"[floor_first] ceil-h adj {_ch_it+1}: h_new={_hn:.3f}m "
                      f"out of range — stop")
                break
            if abs(_hn - float(bounds["height_m"])) < 0.02:
                break   # negligible change
            print(f"[floor_first] ceil-h adj {_ch_it+1}: "
                  f"h={bounds['height_m']:.3f}→{_hn:.3f}m "
                  f"(ceil_v={_cv:.1f} target={_vt:.1f} dv={_dv:+.1f}px)")
            bounds["height_m"] = _hn
            mesh_anchor_ceil = np.array(mesh_anchor_floor, dtype=np.float64)
            mesh_anchor_ceil[1] = float(mesh_anchor_floor[1]) + _hn
            # Re-run zoom phase with updated ceiling anchor (floor pin preserved)
            _zl_dv = float("inf")
            for _zi in range(n_zoom_iter):
                _zp  = np.asarray(cam_pinned["position_m"], dtype=np.float64)
                _zlv = np.asarray(cam_pinned["look_at_m"],  dtype=np.float64)
                _zfv = _normalize(_zlv - _zp)
                if _zfv is None:
                    break
                _zrv = _normalize(np.cross(_zfv, np.array([0., 1., 0.])))
                if _zrv is None:
                    break
                _zuv = np.cross(_zrv, _zfv)
                _zrc = mesh_anchor_ceil - _zp
                _zYc = float(np.dot(_zuv, _zrc))
                _zZc = float(np.dot(_zfv, _zrc))
                if _zZc <= 0.1 or abs(_zYc) < 1e-3:
                    break
                _zfx = W_v / (2. * np.tan(np.radians(hfov_vggt / 2.)))
                _zcy = H_v / 2.
                _zcv = _zcy - _zfx * _zYc / _zZc
                _zdv = _zcv - float(ceil_target_px[1])
                if abs(_zdv) < ceil_tol_px:
                    print(f"[floor_first] ceil-h zoom conv {_zi+1} "
                          f"(dv={_zdv:+.1f}px)")
                    break
                _zsep = max(0.05, 0.01 * _zl_dv)
                if abs(_zdv) > _zl_dv - _zsep:
                    break
                _zl_dv = abs(_zdv)
                _zdn = _zcy - float(ceil_target_px[1])
                if abs(_zdn) < 1e-3:
                    break
                _zZt = _zfx * _zYc / _zdn
                _zdd = float(np.clip(_zZt - _zZc, -max_zoom_step_m, max_zoom_step_m))
                _znp = _zp  - _zdd * _zfv
                _znl = _zlv - _zdd * _zfv
                _ztc = dict(cam_pinned)
                _ztc["position_m"] = _znp.tolist()
                _ztc["look_at_m"]  = _znl.tolist()
                _ztc = _analytical_camera_pin(
                    _ztc,
                    floor_mesh_pt   = mesh_anchor_floor,
                    floor_target_px = floor_target_px,
                )
                _zyt = float(_ztc["position_m"][1])
                if _zyt < min_cam_y_world or _zyt > max_cam_y_world:
                    break
                cam_pinned = _ztc
                print(f"[floor_first] ceil-h zoom {_zi+1}: "
                      f"dv={_zdv:+.1f}px step={_zdd:+.2f}m "
                      f"pos={np.round(_zp - _zdd*_zfv, 3).tolist()}")

    # Honest end-state report so the user can spot scenes that didn't
    # actually align (e.g. when VGGT's pose is geometrically inconsistent
    # with the mesh under our above-floor / below-ceiling constraints).
    final_yaw_d, final_pitch_d = _orbit_residual(cam_pinned)
    final_floor_residual = max(abs(final_yaw_d), abs(final_pitch_d))
    final_ceil_dv: float | None = None
    if ceil_target_px is not None:
        try:
            pos_v   = np.asarray(cam_pinned["position_m"], dtype=np.float64)
            look_v  = np.asarray(cam_pinned["look_at_m"],  dtype=np.float64)
            fwd_v   = look_v - pos_v
            fwd_v   = fwd_v / max(float(np.linalg.norm(fwd_v)), 1e-9)
            right_v = np.cross(fwd_v, np.array([0.0, 1.0, 0.0]))
            right_v = right_v / max(float(np.linalg.norm(right_v)), 1e-9)
            up_v    = np.cross(right_v, fwd_v)
            rel_ceil = mesh_anchor_ceil - pos_v
            Y_ceil   = float(np.dot(up_v,  rel_ceil))
            Z_ceil   = float(np.dot(fwd_v, rel_ceil))
            if Z_ceil > 1e-3:
                fx_r = W_v / (2.0 * np.tan(np.radians(hfov_vggt / 2.0)))
                cy_r = H_v / 2.0
                final_ceil_dv = (cy_r - fx_r * Y_ceil / Z_ceil) - float(ceil_target_px[1])
        except Exception:
            pass

    fully_converged = (
        final_floor_residual < yaw_tol_deg
        and (final_ceil_dv is None or abs(final_ceil_dv) < ceil_tol_px)
    )
    if fully_converged:
        print(f"[floor_first] FULLY CONVERGED: "
              f"yaw={final_yaw_d:+.2f}° pitch={final_pitch_d:+.2f}° "
              f"ceil_dv={final_ceil_dv if final_ceil_dv is None else f'{final_ceil_dv:+.1f}'}px")
    else:
        print(f"[floor_first] WARN: STOPPED WITHOUT FULL CONVERGENCE — "
              f"yaw={final_yaw_d:+.2f}° pitch={final_pitch_d:+.2f}° "
              f"ceil_dv={final_ceil_dv if final_ceil_dv is None else f'{final_ceil_dv:+.1f}'}px "
              f"(scene's VGGT pose may be geometrically inconsistent with the "
              f"mesh under cam.y∈[{min_cam_y_world:.2f}, {max_cam_y_world:.2f}]m)")

    # Final floor pin: scipy polish and ceil-h adjustments both slide the camera,
    # drifting the floor anchor's projected pixel.  One last pin guarantees the
    # anchor lands exactly at floor_target_px regardless of what came before.
    cam_pinned = _analytical_camera_pin(
        cam_pinned,
        floor_mesh_pt   = mesh_anchor_floor,
        floor_target_px = floor_target_px,
    )

    out = dict(cam_pinned)
    # Keep the box-overlay transform fields populated (identity — we project
    # the box through VGGT's camera and the mesh through ours, both at VGGT FOV).
    out["_anchor_transform_R_align"] = np.eye(3).tolist()
    out["_anchor_transform_t_align"] = [0.0, 0.0, 0.0]
    out["_floor_first_converged"]    = bool(fully_converged)
    out["_floor_first_yaw_residual"] = round(float(final_yaw_d), 2)
    out["_floor_first_pitch_residual"] = round(float(final_pitch_d), 2)
    out["_floor_first_mesh_anchor_idx"] = int(mesh_anchor_idx)
    out["_floor_first_mesh_anchor_floor"] = mesh_anchor_floor.tolist()
    if final_ceil_dv is not None:
        out["_floor_first_ceil_dv_px"] = round(float(final_ceil_dv), 1)
    # Store the ceiling Y that the camera was actually calibrated for.
    # This may differ from walls.obj ceiling (which is only updated by
    # post_refine_ceiling if the change exceeds its threshold). The overlay
    # uses this value so the cyan dot matches the calibrated ceiling position.
    out["_calibrated_ceil_y_m"] = round(float(mesh_anchor_ceil[1]), 4)
    if floor_only_position_m is not None and floor_only_look_at_m is not None:
        out["_floor_only_position_m"] = floor_only_position_m
        out["_floor_only_look_at_m"]  = floor_only_look_at_m
    print(f"[floor_first] final: "
          f"pos={np.round(cam_pinned['position_m'], 3).tolist()}  "
          f"look_at={np.round(cam_pinned['look_at_m'], 3).tolist()}")

    # ── Camera-height sanity check ───────────────────────────────────────────
    # If the calibrated cam.y exceeds 75% of ceiling height, VGGT's floor
    # anchor almost certainly landed on furniture rather than the floor-wall
    # junction.  The resulting camera is physically implausible (near the
    # ceiling for a normal eye-level shot).  Discard the VGGT result and
    # fall back to the input VLM camera so downstream stages get something
    # usable rather than a completely wrong pose.
    _ceil_h = float(bounds["height_m"])
    _cam_y  = float(cam_pinned["position_m"][1])
    _sanity_fail = _cam_y > _ceil_h * 0.75
    if _sanity_fail:
        print(f"[floor_first] SANITY FAIL: cam.y={_cam_y:.3f}m is "
              f"{_cam_y / _ceil_h:.0%} of ceiling ({_ceil_h:.3f}m) — "
              f"will try alternate anchor before falling back to VLM camera")
        out["_floor_first_converged"]         = False
        out["_floor_first_cam_y_sanity_fail"] = round(_cam_y, 4)

    # ── Anchor swap retry ───────────────────────────────────────────────────
    # If we ended up pinned against the floor bound with significant
    # remaining residuals AND there's another back-wall mesh corner we
    # haven't tried, the heuristic anchor pick at the top of the function
    # may have selected the wrong corner (the user's hypothesis: "you might
    # have selected the ceiling corner for ground"). Recursively re-run with
    # the alternate anchor and keep whichever attempt produced a more
    # in-bounds, lower-residual camera.
    if (
        _force_mesh_anchor_idx is None
        and len(back_candidates) > 1
        and not fully_converged
    ):
        cam_y_final = float(cam_pinned["position_m"][1])
        cam_x_final = float(cam_pinned["position_m"][0])
        cam_z_final = float(cam_pinned["position_m"][2])
        residual_floor = max(abs(final_yaw_d), abs(final_pitch_d))
        ceil_off = abs(final_ceil_dv) if final_ceil_dv is not None else 0.0
        primary_score = residual_floor + 0.1 * ceil_off
        if _sanity_fail:
            primary_score += 100.0
        # Penalise being pinned to a bound — that's the symptom we're trying
        # to escape via the alt anchor.
        if cam_y_final < min_cam_y_world + 0.1:
            primary_score += 50.0
        if cam_y_final > max_cam_y_world - 0.1:
            primary_score += 50.0
        if cam_x_final < bounds["xmin"] + 0.1:
            primary_score += 50.0
        if cam_x_final > bounds["xmax"] - 0.1:
            primary_score += 50.0
        if cam_z_final < back_z_world:
            primary_score += 50.0
        if cam_z_final > front_z_world + room_depth_z:
            primary_score += 50.0

        alt_idx = next((i for i in back_candidates if i != mesh_anchor_idx), None)
        if alt_idx is not None:
            print(f"[floor_first] primary anchor idx={mesh_anchor_idx} "
                  f"score={primary_score:.2f} (cam.y={cam_y_final:.3f}) — "
                  f"trying alternate anchor idx={alt_idx}")
            try:
                alt_out = _align_floor_first(
                    cam, result,
                    walls_out_dir=walls_out_dir, mesh_path=mesh_path,
                    eye_height_m=eye_height_m,
                    n_yaw_iter=n_yaw_iter,
                    yaw_tol_deg=yaw_tol_deg,
                    _force_mesh_anchor_idx=alt_idx,
                    image_path=image_path,
                )
            except Exception as _alt_e:
                print(f"[floor_first] alternate anchor attempt raised "
                      f"({_alt_e}) — keeping primary")
            else:
                alt_y      = float(alt_out["position_m"][1])
                alt_x      = float(alt_out["position_m"][0])
                alt_yaw    = float(alt_out.get("_floor_first_yaw_residual",   1e9))
                alt_pitch  = float(alt_out.get("_floor_first_pitch_residual", 1e9))
                alt_ceil   = float(alt_out.get("_floor_first_ceil_dv_px",     0.0))
                alt_z      = float(alt_out["position_m"][2])
                alt_score  = max(abs(alt_yaw), abs(alt_pitch)) + 0.1 * abs(alt_ceil)
                if alt_y < min_cam_y_world + 0.1:
                    alt_score += 50.0
                if alt_y > max_cam_y_world - 0.1:
                    alt_score += 50.0
                if alt_x < bounds["xmin"] + 0.1:
                    alt_score += 50.0
                if alt_x > bounds["xmax"] - 0.1:
                    alt_score += 50.0
                if alt_z < back_z_world:
                    alt_score += 50.0
                if alt_z > front_z_world + room_depth_z:
                    alt_score += 50.0
                # Require a meaningful improvement — float ties must not flip the winner
                if alt_score < primary_score - 0.05:
                    print(f"[floor_first] alternate anchor better "
                          f"(score {alt_score:.2f} < {primary_score:.2f}) — "
                          f"using alt result")
                    # If primary had a sanity fail, only use alternate when it
                    # is genuinely clean (no bound violations, score < 50).
                    # Otherwise both cameras are bad — use VLM fallback.
                    if _sanity_fail and alt_score >= 50.0:
                        print(f"[floor_first] SANITY FAIL: alternate also "
                              f"has bound violations (score {alt_score:.1f}) "
                              f"— returning input VLM camera")
                        fallback = dict(cam)
                        for k, v in out.items():
                            if k.startswith("_floor_first_"):
                                fallback[k] = v
                        fallback["_floor_first_converged"] = False
                        return fallback
                    return alt_out
                print(f"[floor_first] primary anchor still wins "
                      f"(score {primary_score:.2f} <= alt {alt_score:.2f})")
    # If primary had a sanity fail and no alternate produced a clean camera
    # (score < 50 means no bound violations), fall back to the VLM camera.
    # A winning score >= 50 means at least one +50 bound penalty fired — the
    # camera is outside the room and should not be used.
    if _sanity_fail:
        # Recompute the winning score to decide whether to trust it.
        _win_y = float(out["position_m"][1])
        _win_x = float(out["position_m"][0])
        _win_z = float(out["position_m"][2])
        _win_score = (max(abs(float(out.get("_floor_first_yaw_residual", 1e9))),
                          abs(float(out.get("_floor_first_pitch_residual", 1e9))))
                      + 0.1 * abs(float(out.get("_floor_first_ceil_dv_px", 0.0))))
        if _win_y < min_cam_y_world + 0.1: _win_score += 50.0
        if _win_y > max_cam_y_world - 0.1: _win_score += 50.0
        if _win_x < bounds["xmin"] + 0.1:  _win_score += 50.0
        if _win_x > bounds["xmax"] - 0.1:  _win_score += 50.0
        if _win_z < back_z_world:           _win_score += 50.0
        if _win_z > front_z_world + room_depth_z: _win_score += 50.0
        if _win_score >= 50.0:
            # Prefer the floor-only intermediate (captured after floor-edges
            # converged, before the ceiling zoom drove the camera out of bounds)
            # over the VLM fallback — it has correct floor-line alignment.
            _fo_pos  = out.get("_floor_only_position_m")
            _fo_look = out.get("_floor_only_look_at_m")
            if _fo_pos is not None and _fo_look is not None:
                _foy = float(_fo_pos[1])
                _fox = float(_fo_pos[0])
                _foz = float(_fo_pos[2])
                _fo_in_bounds = (
                    min_cam_y_world <= _foy <= max_cam_y_world
                    and bounds["xmin"] <= _fox <= bounds["xmax"]
                    and back_z_world <= _foz <= front_z_world + room_depth_z
                )
                if _fo_in_bounds:
                    print(f"[floor_first] SANITY FAIL: using floor-only "
                          f"intermediate pos={[round(v,3) for v in _fo_pos]} "
                          f"(ceiling zoom caused out-of-bounds)")
                    fo_cam = dict(out)
                    fo_cam["position_m"] = _fo_pos
                    fo_cam["look_at_m"]  = _fo_look
                    fo_cam["_floor_first_converged"] = False
                    fo_cam["_floor_first_used_floor_only"] = True
                    return fo_cam
            print(f"[floor_first] SANITY FAIL: winner score={_win_score:.1f} "
                  f"still has bound violations — returning input VLM camera")
            fallback = dict(cam)
            for k, v in out.items():
                if k.startswith("_floor_first_"):
                    fallback[k] = v
            fallback["_floor_first_converged"] = False
            return fallback
        print(f"[floor_first] SANITY FAIL resolved by alternate "
              f"(winner score={_win_score:.2f})")
    return out


def _pin_look_at_to_anchor_pixel(
    cam: dict,
    mesh_anchor: np.ndarray,
    target_u: float,
    target_v: float,
) -> dict:
    """
    Adjust look_at so mesh_anchor (back-wall floor corner) projects to
    (target_u, target_v) in the render frame.  Camera position is unchanged.

    Horizontal (yaw): architectural snap formula — look_x chosen so anchor_x
      appears at target_u column.
    Vertical (tilt):  look_y chosen so a floor point (Y=0) at back-wall depth
      appears at target_v row, using the architectural tilt relation
      yc_floor ≈ −look_y when zc ≈ cam_z.
    """
    cam   = dict(cam)
    pos   = np.array(cam["position_m"], dtype=np.float64)
    W     = int(cam["width_px"]); H = int(cam["height_px"])
    hfov  = float(cam["hfov_deg"])
    fx    = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    cam_x, cam_y, cam_z = float(pos[0]), float(pos[1]), float(pos[2])
    anchor_x = float(mesh_anchor[0])
    anchor_y = float(mesh_anchor[1])   # floor Y (typically 0)

    # Horizontal pin
    r = (anchor_x - cam_x) / max(cam_z, 0.1)
    k = (target_u - cx) / fx
    denom = 1.0 + r * k
    look_x = (cam_x + cam_z * (r - k) / denom) if abs(denom) > 1e-6 else cam_x

    # Vertical pin: exact formula using the anchor's true projection depth zc.
    #
    # Render uses architectural (two-point) perspective:
    #   tilt_tan = (look_y - cam_y) / d_horiz     where d_horiz = ||(look-pos)_xz||
    #   yc       = (anchor_y - cam_y) - tilt_tan * zc
    #   v        = cy - fx * yc / zc
    #
    # Solving for look_y:
    #   yc_target = -(target_v - cy) * zc / fx
    #   tilt_tan  = ((anchor_y - cam_y) - yc_target) / zc
    #   look_y    = cam_y + tilt_tan * d_horiz
    #
    # zc is the anchor's depth along the horizontal forward — computed from look_x
    # (already pinned above) so the horizontal direction is fixed.
    _look_z = 0.0  # back wall
    _fwd_h_v = np.array([look_x - cam_x, 0.0, _look_z - cam_z], dtype=np.float64)
    _fwd_h_n = float(np.linalg.norm(_fwd_h_v))
    if _fwd_h_n > 1e-6:
        _fwd_h_u = _fwd_h_v / _fwd_h_n          # unit horizontal forward
        _d_anchor = np.array([anchor_x - cam_x,
                               anchor_y - cam_y,
                               _look_z - cam_z], dtype=np.float64)
        _zc = float(np.dot(_d_anchor, _fwd_h_u))  # true depth of anchor
        if _zc > 0.1:
            _yc_target  = -(target_v - cy) * _zc / fx
            _tilt_tan   = ((anchor_y - cam_y) - _yc_target) / _zc
            look_y = float(cam_y + _tilt_tan * _fwd_h_n)
        else:
            look_y = (target_v - cy) * cam_z / fx  # fallback
    else:
        look_y = (target_v - cy) * cam_z / fx      # degenerate: looking straight up/down
    # No upper cap on look_y: for low cameras (cam_y near floor), look_y > cam_y
    # is valid and means tilting slightly upward — which correctly pushes a
    # below-camera floor anchor further down in the image.
    look_y = float(np.clip(look_y, -cam_y, cam_y + 2.0))

    old = cam["look_at_m"]
    cam["look_at_m"] = [round(look_x, 4), round(look_y, 4), 0.0]
    print(f"[align] look_at pin: ({old[0]:.3f},{old[1]:.3f},{old[2]:.3f}) → "
          f"({look_x:.3f},{look_y:.3f},0.0)  "
          f"[anchor_x={anchor_x:.2f}m → px ({target_u:.0f},{target_v:.0f})]")
    return cam


# ── main alignment entry point ────────────────────────────────────────────────

def align_camera(
    vggt_out_dir: str,
    walls_out_dir: str,
    frame: int = 0,
    subsample: int = 4,
    n_iter: int = _N_ITER_DEFAULT,
) -> dict:
    """
    Align the VLM camera to the VGGT Manhattan box via iterative VLM calibration.

    Returns the final camera dict (same schema as camera.json / camera_vggt.json).
    """
    out_root = Path(walls_out_dir)

    # ── 1. Manhattan estimation ───────────────────────────────────────────────
    # Manhattan fitting uses VGGT's native depth_0.npy + normal_0.npy.  VGGT
    # depth is metric (metres) so the plane positions and cam-Z values are
    # meaningful for corner selection; a relative depth map would collapse both
    # back corners to the same cam-Z and make the anchor selection arbitrary.
    print("[align] Running Manhattan estimation …")
    result = estimate_manhattan(vggt_out_dir, frame=frame, subsample=subsample)
    fx_v, fy_v, W_px, H_px = _load_vggt_intrinsics(vggt_out_dir, frame)
    hfov_vggt = float(np.degrees(2.0 * np.arctan(W_px / (2.0 * fx_v))))
    vfov_vggt = float(np.degrees(2.0 * np.arctan(H_px / (2.0 * fy_v))))

    # ── 2. Anchor extraction ──────────────────────────────────────────────────
    floor_w, ceil_w = _find_back_anchor(result)

    # ── 3. Static visualizations ──────────────────────────────────────────────
    # Load source image path early — needed for ceiling visibility check and overlays
    src_image = _load_vggt_image_path(vggt_out_dir, frame)

    # Ceiling visibility: ask the VLM about the real reference photo, not
    # the Manhattan projection. This decides whether the alignment loop should
    # constrain both floor + ceiling anchors or floor only.
    ceil_vis = (
        _detect_ceiling_visible(src_image)
        if src_image and Path(src_image).exists()
        else True   # fallback: assume visible when image unavailable
    )

    # manhattan_reference.png — blank background + box + anchors (VLM target)
    ref_img_path = str(out_root / "manhattan_reference.png")
    render_manhattan_reference(result, floor_w, ceil_w, ref_img_path)

    # manhattan_vggt.png — same clean box + anchors as the reference image,
    # but drawn over the source photo instead of a white background. Keeps
    # the visual style consistent with manhattan_reference.png.
    if src_image and Path(src_image).exists():
        try:
            from PIL import Image as _PIL
            _photo = _PIL.open(src_image).convert("RGB").resize(
                (int(result["W"]), int(result["H"])), _PIL.LANCZOS)
            _draw = ImageDraw.Draw(_photo)
            _draw_box_and_anchors(_draw, result, floor_w, ceil_w, scale=1.0)
            _photo.save(str(out_root / "manhattan_vggt.png"))
            print(f"[align] Manhattan over photo → "
                  f"{out_root / 'manhattan_vggt.png'}")
        except Exception as _e:
            print(f"[align] manhattan_vggt overlay failed ({_e}); "
                  f"falling back to visualize_manhattan")
            visualize_manhattan(
                vggt_out_dir=vggt_out_dir,
                image_path=src_image,
                out_path=str(out_root / "manhattan_vggt.png"),
                frame=frame,
                subsample=subsample,
            )

    # ── 4. Starting camera from VLM ───────────────────────────────────────────
    vlm_cam   = _load_vlm_camera(walls_out_dir)
    width_px  = int(vlm_cam.get("width_px",  W_px))
    height_px = int(vlm_cam.get("height_px", H_px))

    cam: dict = dict(vlm_cam)
    # Keep Stage 1 VLM's FOV — VGGT's estimated FOV is for the reference photo
    # lens and does not match the render camera position, causing heavy zoom.
    # VGGT FOV is stored as metadata only (see _vggt_fov_h/v_deg below).
    cam["width_px"]  = width_px
    cam["height_px"] = height_px
    cam["up"]        = [0.0, 1.0, 0.0]

    print(f"[align] Keeping Stage-1 FOV: hfov={cam.get('hfov_deg')}°  "
          f"(VGGT estimated: {hfov_vggt:.1f}°)")

    # ── 5. Anchor-transform calibration + multi-point refinement ────────────
    # 5a. Closed-form anchor pin: rigid transform T (R + t) from VGGT world
    # to mesh world that takes the deepest box floor corner to the mesh's
    # back-wall anchor corner. After this step the green/cyan back-vertical-
    # edge anchors are pinned exactly, but the floor-wall lines radiating
    # from the anchor can still be off (mesh aspect ratio + camera yaw).
    # 5b. Multi-point optimizer: refine cam pos + look-at against all 4
    # floor + 4 ceiling corner pixel targets (heavy weight on the anchor
    # to preserve the pin) so the floor-wall lines also reproject onto the
    # VGGT box edges.
    mesh_path = None
    try:
        mesh_path = _find_wall_mesh(walls_out_dir)
    except FileNotFoundError:
        print("[align] No wall mesh — skipping anchor-transform calibration.")

    if mesh_path is not None:
        from floorplan.wall_line.render_room import render_room

        # ── Pre-rotate seed camera if VGGT anchor is off-screen ──────────────
        # If the stage-1 camera yaw is badly wrong (e.g. Stage 1 picked the
        # wrong back-wall corner), the VGGT floor anchor will project off-screen
        # and _align_floor_first cannot converge.  Detect this and reorient the
        # camera to face the back-wall anchor region before alignment starts.
        # Also save anchor data for the post-floor-first fallback pin below.
        _saved_mesh_anchor: np.ndarray | None = None
        _saved_anchor_target_u: float | None  = None
        _saved_anchor_target_v: float | None  = None
        try:
            _W_r = int(cam["width_px"]); _H_r = int(cam["height_px"])
            _hfov_r = float(cam.get("hfov_deg", 70))
            _vfov_r = float(cam.get("vfov_deg", _hfov_r))
            _fxr = _W_r / (2.0 * np.tan(np.radians(_hfov_r / 2.0)))
            _fyr = _H_r / (2.0 * np.tan(np.radians(_vfov_r / 2.0)))
            _K_v = result["K"]
            _fxv = float(_K_v[0, 0]); _fyv = float(_K_v[1, 1])
            _cxv = float(_K_v[0, 2]); _cyv = float(_K_v[1, 2])

            _floor_px = _project_pt(floor_w, result["K"], result["R"], result["t"])
            if _floor_px is not None:
                # VGGT floor anchor mapped to render frame pixel coordinates
                _saved_anchor_target_u = (float(_floor_px[0]) - _cxv) * (_fxr / _fxv) + _W_r / 2.0
                _saved_anchor_target_v = (float(_floor_px[1]) - _cyv) * (_fyr / _fyv) + _H_r / 2.0

                # Find the mesh anchor corner closest to the VGGT floor anchor pixel
                _mesh_anch = _find_anchor_mesh_corner(
                    mesh_path, cam, _floor_px, int(result["W"]), int(result["H"]))
                if _mesh_anch is not None:
                    _saved_mesh_anchor = np.asarray(_mesh_anch, dtype=np.float64)
                    # Project it through the current render camera
                    _anch_px = _project_render(_mesh_anch, cam)
                    _off_screen = (
                        _anch_px is None
                        or _anch_px[0] < 0 or _anch_px[0] >= _W_r
                        or _anch_px[1] < 0 or _anch_px[1] >= _H_r
                    )
                    if _off_screen:
                        # Rotate seed camera (horizontal only) to face the anchor
                        _pos = np.array(cam["position_m"], dtype=np.float64)
                        _dir = np.array(_mesh_anch, dtype=np.float64) - _pos
                        _dir[1] = 0.0   # keep horizontal
                        _dir_n = float(np.linalg.norm(_dir))
                        if _dir_n > 0.1:
                            _unit = _dir / _dir_n
                            _look_dist = max(float(np.linalg.norm(
                                np.array(cam["look_at_m"]) - _pos)), 0.5)
                            _new_look = _pos + _unit * _look_dist
                            print(f"[align] Anchor off-screen → pre-rotate seed camera: "
                                  f"look_at {cam['look_at_m']} → "
                                  f"{np.round(_new_look, 3).tolist()}")
                            cam["look_at_m"] = _new_look.tolist()
        except Exception as _pre_e:
            print(f"[align] Pre-rotation check failed ({_pre_e}) — skipping")

        try:
            cam = _align_floor_first(
                cam, result, walls_out_dir=out_root, mesh_path=mesh_path,
                eye_height_m=1.4,
                image_path=src_image,
            )
        except Exception as _ff_e:
            print(f"[floor_first] unexpected failure ({_ff_e}) — keeping VLM camera")

        # ── Post-alignment fallback: look_at pin ─────────────────────────────────
        # If the alternate anchor won, _align_floor_first calibrated for a
        # different mesh corner than _saved_mesh_anchor (the primary corner).
        # Update _saved_mesh_anchor to the winning anchor's 3D point so the
        # look_at pin below checks the correct corner — otherwise the primary
        # corner projects 1000+px off, the pin rotates the camera 60°+, and
        # the alternate anchor's calibration is silently discarded.
        _winning_anchor = cam.get("_floor_first_mesh_anchor_floor")
        if _winning_anchor is not None and _saved_mesh_anchor is not None:
            _wa = np.asarray(_winning_anchor, dtype=np.float64)
            if not np.allclose(_wa, _saved_mesh_anchor, atol=0.01):
                print(f"[align] Updating saved mesh anchor from "
                      f"{np.round(_saved_mesh_anchor, 3).tolist()} → "
                      f"{np.round(_wa, 3).tolist()} (alternate anchor won)")
                _saved_mesh_anchor = _wa

        # Recompute the target pixel HERE using the *current* cam's FOV — not the
        # pre-rotation-time value in _saved_anchor_target_u/v.  _align_floor_first
        # may change hfov_deg (to VGGT's value), so the pre-saved target is in the
        # wrong pixel space.  Recomputing ensures the target and pin are consistent.
        if _saved_mesh_anchor is not None:
            try:
                _K_pin  = result["K"]
                _fxv_p  = float(_K_pin[0, 0]); _fyv_p = float(_K_pin[1, 1])
                _cxv_p  = float(_K_pin[0, 2]); _cyv_p = float(_K_pin[1, 2])
                _W_p    = int(cam["width_px"]); _H_p   = int(cam["height_px"])
                _hfov_p = float(cam.get("hfov_deg", 70.0))
                _vfov_p = float(cam.get("vfov_deg", _hfov_p))
                _fxr_p  = _W_p / (2.0 * np.tan(np.radians(_hfov_p / 2.0)))
                _fyr_p  = _H_p / (2.0 * np.tan(np.radians(_vfov_p / 2.0)))
                _fl_pin = _project_pt(floor_w, result["K"], result["R"], result["t"])
                if _fl_pin is None:
                    raise ValueError("floor_w behind VGGT camera")
                _tgt_u = (float(_fl_pin[0]) - _cxv_p) * (_fxr_p / _fxv_p) + _W_p / 2.0
                _tgt_v = (float(_fl_pin[1]) - _cyv_p) * (_fyr_p / _fyv_p) + _H_p / 2.0

                _cur_px = _project_render(_saved_mesh_anchor, cam)
                _err    = float("inf")
                if _cur_px is not None:
                    _err = float(np.hypot(_cur_px[0] - _tgt_u, _cur_px[1] - _tgt_v))
                print(f"[align] Post-floor-first anchor residual: {_err:.1f}px  "
                      f"(target=({_tgt_u:.0f},{_tgt_v:.0f})  "
                      f"current={None if _cur_px is None else (round(_cur_px[0]),round(_cur_px[1]))})")
                if _err > 50:
                    cam = _pin_look_at_to_anchor_pixel(cam, _saved_mesh_anchor,
                                                       _tgt_u, _tgt_v)
            except Exception as _pin_e:
                print(f"[align] Post-alignment pin failed ({_pin_e}) — skipping")

        # Final render → render_vggt.png  (+ FOV coverage extension if needed)
        _bg = (40, 40, 40)
        final_render_path = str(out_root / "render_vggt.png")
        final_cam_tmp = str(out_root / "_final_cam_tmp.json")
        Path(final_cam_tmp).write_text(json.dumps(cam, indent=2))
        try:
            render_room(
                mesh_path=str(mesh_path),
                camera_json_path=final_cam_tmp,
                out_path=final_render_path,
                texture_dir=None,
                bg_color=_bg,
            )
            print(f"[align] Final render → {final_render_path}")

            # If background is visible at left/right edges, extend mesh and re-render
            new_depth = _extend_mesh_for_fov_coverage(
                mesh_path, cam, final_render_path, bg_color=_bg
            )
            if new_depth is not None:
                # Propagate new room depth into wall_context so downstream
                # stages (texturing, feature placement) use the updated geometry
                wctx = cam.setdefault("wall_context", {})
                for side in ("left", "right"):
                    if side in wctx:
                        wctx[side]["length_m"] = round(new_depth, 3)
                # Re-write the tmp camera file with updated wall_context
                Path(final_cam_tmp).write_text(json.dumps(cam, indent=2))
                # Also propagate to the persistent files placement code reads:
                #   * floorplan_analysis.json — `_get_room_dims` checks this FIRST
                #   * camera.json             — fallback path for room dims
                # Without this, `place_furniture_vggt` plans positions inside the
                # pre-extension room (4.0m here) while the rendered mesh is the
                # extended one (4.82m), leaving an unused band where init_slide
                # / collision-resolve / final_anchor measurements are all off.
                _persist_root = Path(walls_out_dir)
                _new_depth_r = round(float(new_depth), 3)
                _fp_path = _persist_root / "floorplan_analysis.json"
                if _fp_path.exists():
                    try:
                        _fp_data = json.loads(_fp_path.read_text())
                        _fp_data.setdefault("room", {})["floor_depth_m"] = _new_depth_r
                        for _w in _fp_data.get("walls", []):
                            if _w.get("orientation") in ("left", "right"):
                                _w["length_m"] = _new_depth_r
                        _fp_path.write_text(json.dumps(_fp_data, indent=2))
                        print(f"[align] floorplan_analysis.json depth → {_new_depth_r} m")
                    except Exception as _fpe:
                        print(f"[align] floorplan_analysis.json update failed: {_fpe}")
                _cam_persist = _persist_root / "camera.json"
                if _cam_persist.exists():
                    try:
                        _cdata = json.loads(_cam_persist.read_text())
                        _cwctx = _cdata.setdefault("wall_context", {})
                        for _side in ("left", "right"):
                            if _side in _cwctx:
                                _cwctx[_side]["length_m"] = _new_depth_r
                        _cam_persist.write_text(json.dumps(_cdata, indent=2))
                        print(f"[align] camera.json wall_context.left/right.length_m → {_new_depth_r} m")
                    except Exception as _ce:
                        print(f"[align] camera.json update failed: {_ce}")
                render_room(
                    mesh_path=str(mesh_path),
                    camera_json_path=final_cam_tmp,
                    out_path=final_render_path,
                    texture_dir=None,
                    bg_color=_bg,
                )
                print(f"[align] Re-render after mesh extension → {final_render_path}")
            # Manhattan overlay — use actual mesh corner 3D points for the
            # anchor dots so they land exactly where the rendered corners appear.
            # Overlay: both dots and wireframe from VGGT Manhattan projection,
            # FOV-corrected to the render frame so they are consistent with each
            # other.  Passing render_cam enables use_fov_map=True which applies
            # (fx_r/fx_v) scaling instead of simple pixel scaling, fixing the
            # ~50px offset that appears when VGGT FOV ≠ render FOV.
            render_manhattan_overlay(
                final_render_path, result, floor_w, ceil_w,
                str(out_root / "render_vggt_overlay.png"),
                render_cam=cam,
            )
        except Exception as e:
            print(f"[align] Final render failed: {e}")
        finally:
            Path(final_cam_tmp).unlink(missing_ok=True)

    # X-clamp removed: the final floor pin inside _align_floor_first may place
    # the camera slightly outside the room on the X axis to achieve pixel-exact
    # floor anchor alignment. Back-face culling in render_room handles cameras
    # outside the room boundary without rendering artifacts for typical views.
    # Clamping here would undo the floor pin and re-introduce pixel error.

    # ── 7. Anchor pixel metadata (debug) ─────────────────────────────────────
    # Store anchor positions in render-camera pixel coordinates (FOV-corrected).
    _K  = result["K"]
    _fl = _project_pt(floor_w, _K, result["R"], result["t"])
    _ce = _project_pt(ceil_w,  _K, result["R"], result["t"])
    _fxv = float(_K[0, 0]); _fyv = float(_K[1, 1])
    _cxv = float(_K[0, 2]); _cyv = float(_K[1, 2])
    _hfr = float(cam.get("hfov_deg", 70.0)); _vfr = float(cam.get("vfov_deg", _hfr))
    _Wr  = int(cam["width_px"]);  _Hr  = int(cam["height_px"])
    _fxr = _Wr / (2.0 * np.tan(np.radians(_hfr / 2.0)))
    _fyr = _Hr / (2.0 * np.tan(np.radians(_vfr / 2.0)))
    def _to_rpx(px):
        u = (float(px[0]) - _cxv) * (_fxr / _fxv) + _Wr / 2.0
        v = (float(px[1]) - _cyv) * (_fyr / _fyv) + _Hr / 2.0
        return [round(u, 1), round(v, 1)]
    cam["_floor_anchor_px"] = _to_rpx(_fl) if _fl is not None else None
    cam["_ceil_anchor_px"]  = _to_rpx(_ce) if _ce is not None else None
    cam["_vggt_fov_h_deg"] = round(hfov_vggt, 2)
    cam["_vggt_fov_v_deg"] = round(vfov_vggt, 2)
    if "wall_context" in vlm_cam:
        cam["wall_context"] = vlm_cam["wall_context"]

    return cam


# ── Stage-1-free entry point ──────────────────────────────────────────────────

def _parse_mesh_bounds(mesh_path: str) -> dict:
    """Read walls.obj and return the vertex bounding box."""
    xs, ys, zs = [], [], []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    xs.append(float(parts[1]))
                    ys.append(float(parts[2]))
                    zs.append(float(parts[3]))
    return {
        "xmin": min(xs), "xmax": max(xs),
        "ymin": min(ys), "ymax": max(ys),
        "zmin": min(zs), "zmax": max(zs),
        "width_m":  round(max(xs) - min(xs), 3),
        "height_m": round(max(ys) - min(ys), 3),
        "depth_m":  round(max(zs) - min(zs), 3),
    }


def _initial_camera_from_bounds(
    mesh_bounds: dict,
    img_w: int = 1024,
    img_h: int = 1024,
    hfov_deg: float = 70.0,
) -> dict:
    """
    Reliable heuristic initial camera from mesh bounding box.

    Position: centre in X, 45 % of room height in Y, 80 % of room depth in Z
    (camera near the far Z wall, looking toward the back wall at Z_min).
    FOV is fixed at hfov_deg (default 70 °) — never estimated by VLM because
    VLMs consistently under-estimate FOV (too telephoto) when asked.
    """
    B     = mesh_bounds
    cx    = (B["xmin"] + B["xmax"]) / 2.0
    H     = B["height_m"]
    back_z = B["zmin"]
    cam_z  = B["zmax"]
    return {
        "position_m": [cx,  H * 0.45,  cam_z * 0.80],
        "look_at_m":  [cx,  H * 0.45,  back_z],
        "up":         [0.0, 1.0, 0.0],
        "hfov_deg":   hfov_deg,
        "width_px":   img_w,
        "height_px":  img_h,
    }


def align_camera_direct(
    vggt_out_dir: str,
    walls_out_dir: str,
    frame: int = 0,
    subsample: int = 4,
    n_iter: int = _N_ITER_DEFAULT,
) -> dict:
    """
    Align the render camera to the VGGT Manhattan box WITHOUT Stage 1.

    Replaces the Stage 1 VLM scene-analysis camera with a Manhattan-guided
    VLM initial placement, then refines iteratively with the two-phase
    corner-pixel + orbit approach.

    Pipeline
    --------
    1. Manhattan estimation → anchor corners, reference image
    2. VLM places initial camera from Manhattan wireframe + mesh bounds
    3. Iterative refinement:
         render → overlay Manhattan → VLM corner offset + orbit angles
         → translate to pin corner → orbit to align lines
    4. Final render saved as render_vggt.png

    Returns the final camera dict.
    """
    out_root = Path(walls_out_dir)

    # ── 1. Manhattan estimation ───────────────────────────────────────────────
    print("[align-direct] Running Manhattan estimation …")
    result = estimate_manhattan(vggt_out_dir, frame=frame, subsample=subsample)
    _, _, W_px, H_px = _load_vggt_intrinsics(vggt_out_dir, frame)

    # ── 2. Anchor extraction ──────────────────────────────────────────────────
    floor_w, ceil_w = _find_back_anchor(result)

    # ── 3. Reference visualizations ───────────────────────────────────────────
    src_image = _load_vggt_image_path(vggt_out_dir, frame)
    ceil_vis  = (
        _detect_ceiling_visible(src_image)
        if src_image and Path(src_image).exists()
        else True
    )

    ref_img_path = str(out_root / "manhattan_reference.png")
    render_manhattan_reference(result, floor_w, ceil_w, ref_img_path)

    if src_image and Path(src_image).exists():
        visualize_manhattan(
            vggt_out_dir=vggt_out_dir,
            image_path=src_image,
            out_path=str(out_root / "manhattan_vggt.png"),
            frame=frame,
            subsample=subsample,
        )

    # ── 4. Anchor fractions (in render-camera pixel space) ───────────────────
    K_v, R_v, t_v = result["K"], result["R"], result["t"]
    fl_px_pre = _project_pt(floor_w, K_v, R_v, t_v)
    ce_px_pre = _project_pt(ceil_w,  K_v, R_v, t_v)
    # Convert VGGT pixels → render pixels so the VLM hint reflects the render
    # FOV (70° hfov) rather than the VGGT reference FOV.
    _fyv_d = float(K_v[1, 1]);  _cyv_d = float(K_v[1, 2])
    _fyr_d = W_px / (2.0 * np.tan(np.radians(35.0)))  # fy for 70° hfov, square px
    def _v_to_render(v_vggt: float) -> float:
        return (v_vggt - _cyv_d) * (_fyr_d / _fyv_d) + H_px / 2.0
    floor_frac = (
        _v_to_render(float(fl_px_pre[1])) / H_px if fl_px_pre is not None else 0.7
    )
    ceil_frac = (
        _v_to_render(float(ce_px_pre[1])) / H_px if ce_px_pre is not None else 0.1
    )

    # ── 5. Initial camera from VLM (no Stage 1 dependency) ───────────────────
    mesh_path   = _find_wall_mesh(walls_out_dir)
    mesh_bounds = _parse_mesh_bounds(str(mesh_path))
    print(f"[align-direct] Mesh bounds: "
          f"X={mesh_bounds['xmin']:.1f}–{mesh_bounds['xmax']:.1f}  "
          f"Y={mesh_bounds['ymin']:.1f}–{mesh_bounds['ymax']:.1f}  "
          f"Z={mesh_bounds['zmin']:.1f}–{mesh_bounds['zmax']:.1f}")

    cam = _initial_camera_from_bounds(mesh_bounds, img_w=W_px, img_h=H_px)
    print(f"[align-direct] Initial camera: pos={cam['position_m']}  "
          f"look={cam['look_at_m']}  hfov={cam['hfov_deg']}°")

    # ── 6. Iterative two-phase refinement ─────────────────────────────────────
    from floorplan.wall_line.render_room import render_room

    initial_pos = np.array(cam["position_m"], dtype=np.float64)

    for it in range(n_iter):
        print(f"\n[align-direct] ── Iteration {it + 1}/{n_iter} ──────────")

        render_out_path = str(out_root / f"render_iter_{it}.png")
        tmp_cam_path    = str(out_root / f"_tmp_cam_{it}.json")
        Path(tmp_cam_path).write_text(json.dumps(cam, indent=2))
        try:
            render_room(
                mesh_path=str(mesh_path),
                camera_json_path=tmp_cam_path,
                out_path=render_out_path,
                texture_dir=None,
                bg_color=(40, 40, 40),
            )
        except Exception as e:
            print(f"[align-direct] Render failed at iter {it}: {e}")
            break
        finally:
            Path(tmp_cam_path).unlink(missing_ok=True)

        overlay_path = str(out_root / f"render_overlay_{it}.png")
        render_manhattan_overlay(render_out_path, result, floor_w, ceil_w, overlay_path)

        adj      = _call_vlm_for_camera_adjustment(
            ref_img_path, overlay_path,
            ceil_visible=ceil_vis,
            floor_frac=floor_frac,
            ceil_frac=ceil_frac,
        )
        du, dv   = adj.get("corner_offset_px", [0, 0])
        line_yaw = float(adj.get("line_yaw_delta_deg",   0.0))
        line_pit = float(adj.get("line_pitch_delta_deg", 0.0))

        if abs(du) < 5 and abs(dv) < 5 and abs(line_yaw) < 0.5 and abs(line_pit) < 0.5:
            print(f"[align-direct] Converged at iteration {it + 1}.")
            break

        cam = _apply_camera_corner_orbit(
            cam,
            corner_du=float(du), corner_dv=float(dv),
            line_yaw_deg=line_yaw, line_pitch_deg=line_pit,
            initial_pos=initial_pos,
        )
        print(f"[align-direct] pos={np.round(cam['position_m'], 3)}  "
              f"look={np.round(cam['look_at_m'], 3)}")

    for it in range(n_iter):
        (out_root / f"_tmp_cam_{it}.json").unlink(missing_ok=True)

    # Final render
    final_tmp = str(out_root / "_final_cam_tmp.json")
    Path(final_tmp).write_text(json.dumps(cam, indent=2))
    try:
        render_room(
            mesh_path=str(mesh_path),
            camera_json_path=final_tmp,
            out_path=str(out_root / "render_vggt.png"),
            texture_dir=None,
            bg_color=(40, 40, 40),
        )
        print(f"[align-direct] Final render → {out_root / 'render_vggt.png'}")
    except Exception as e:
        print(f"[align-direct] Final render failed: {e}")
    finally:
        Path(final_tmp).unlink(missing_ok=True)

    # Anchor pixel metadata — store in render-camera pixel space (FOV-corrected).
    _fxv_dm = float(K_v[0, 0]);  _cxv_dm = float(K_v[0, 2])
    _fxr_dm = W_px / (2.0 * np.tan(np.radians(35.0)))  # fx for 70° hfov
    def _rpx_d(px):
        u = (float(px[0]) - _cxv_dm) * (_fxr_dm / _fxv_dm) + W_px / 2.0
        v = (float(px[1]) - _cyv_d)  * (_fyr_d  / _fyv_d)  + H_px / 2.0
        return [round(u, 1), round(v, 1)]
    cam["_floor_anchor_px"] = _rpx_d(fl_px_pre) if fl_px_pre is not None else None
    cam["_ceil_anchor_px"]  = _rpx_d(ce_px_pre) if ce_px_pre is not None else None

    return cam


# ── unchanged utilities ───────────────────────────────────────────────────────

def save_camera(camera: dict, out_path: str) -> str:
    Path(out_path).write_text(json.dumps(camera, indent=2))
    print(f"[align] Saved camera → {out_path}")
    return str(out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(
    vggt_out_dir: str,
    walls_out_dir: str,
    out_name: str = "camera_vggt.json",
    frame: int = 0,
    subsample: int = 4,
    n_iter: int = _N_ITER_DEFAULT,
) -> str:
    cam  = align_camera(vggt_out_dir, walls_out_dir,
                        frame=frame, subsample=subsample, n_iter=n_iter)
    dest = str(Path(walls_out_dir) / out_name)
    save_camera(cam, dest)
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Align VLM camera with VGGT Manhattan box via iterative VLM loop"
    )
    ap.add_argument("--vggt-out",  required=True,
                    help="VGGT output dir (camera.json, depth_0.npy, normal_0.npy)")
    ap.add_argument("--walls-out", required=True,
                    help="Walls output dir (walls.obj, camera.json)")
    ap.add_argument("--out-name",  default="camera_vggt.json")
    ap.add_argument("--frame",     type=int, default=0)
    ap.add_argument("--subsample", type=int, default=4)
    ap.add_argument("--n-iter",    type=int, default=_N_ITER_DEFAULT,
                    help=f"VLM calibration iterations (default {_N_ITER_DEFAULT})")
    args = ap.parse_args()
    main(args.vggt_out, args.walls_out, args.out_name,
         args.frame, args.subsample, args.n_iter)
