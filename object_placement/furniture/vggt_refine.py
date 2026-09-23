"""VGGT-based ground-truth position refinement for furniture placement.

Companion to `place_furniture_vggt.py`. The idea: the photo's VGGT depth map
gives an approximate metric depth for every photo pixel. For a placed
furniture item we know its segmentation mask in photo coords; back-projecting
the masked pixels through the HARMONY camera + photo VGGT depth (and
applying a once-computed Sim(3) into world meters) yields a "what does the
real photo think this object's position is" prediction. We blend that with
the silhouette mask_align position so the post-VLM placement honours both.

Method B (global-calib) is used: the Sim(3) is computed once at run start
from the *empty floorplan render* and reused for every furniture entry.
This is dramatically cheaper than re-running VGGT per placement and the
calibration is dominated by walls/floor (which don't change as we add
furniture), so it stays valid throughout the run.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Ablation: suppress the large per-iteration vggt_refine visualization output
# (depth_alignment/vggt_refine_viz) to keep disk usage bounded across many runs.
_ABL_NO_VIZ = os.environ.get("SCENEWEAVE_ABLATE_NO_VIZ") == "1"

import numpy as np
from PIL import Image

# ─── Camera helpers ────────────────────────────────────────────────────────────

def _camera_to_KE(cam: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    """HARMONY look-at spec → (K 3x3, extrinsic 3x4, W, H), OpenCV convention."""
    pos     = np.array(cam["position_m"], dtype=np.float64)
    look_at = np.array(cam["look_at_m"],  dtype=np.float64)
    up_w    = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    W, H    = int(cam["width_px"]), int(cam["height_px"])
    hfov    = float(cam["hfov_deg"])

    fwd = look_at - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_w); right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    R = np.stack([right, -cam_up, fwd], axis=0)
    t = -R @ pos
    extrinsic = np.concatenate([R, t[:, None]], axis=1).astype(np.float32)

    fx = W / (2.0 * math.tan(math.radians(hfov / 2.0)))
    K = np.array([[fx, 0.0, W/2.0],
                  [0.0, fx, H/2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, extrinsic, W, H


def _unproject_depth(depth: np.ndarray, K: np.ndarray, extrinsic_3x4: np.ndarray,
                     pixel_mask: np.ndarray | None = None) -> np.ndarray:
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if pixel_mask is None:
        ys, xs = np.mgrid[:H, :W]
        ys, xs = ys.ravel(), xs.ravel()
        d = depth.ravel()
    else:
        ys, xs = np.where(pixel_mask)
        d = depth[ys, xs]
    valid = (d > 0) & np.isfinite(d)
    ys, xs, d = ys[valid], xs[valid], d[valid]
    cam_x = (xs - cx) * d / fx
    cam_y = (ys - cy) * d / fy
    cam_z = d
    cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1).astype(np.float64)
    R = extrinsic_3x4[:3, :3].astype(np.float64); t = extrinsic_3x4[:3, 3].astype(np.float64)
    return ((cam_pts - t) @ R).astype(np.float32)


# ─── Robust median + Sim(3) alignment ──────────────────────────────────────────

def _dominant_centroid(points: np.ndarray) -> tuple[np.ndarray, int]:
    """Iterative-median + MAD outlier rejection. RAM-safe."""
    if points.shape[0] == 0:
        return np.array([np.nan]*3, dtype=np.float32), 0
    if points.shape[0] > 200_000:
        idx = np.random.default_rng(0).choice(points.shape[0], 200_000, replace=False)
        pts = points[idx].astype(np.float64)
    else:
        pts = points.astype(np.float64)
    inliers = pts
    for _ in range(5):
        med  = np.median(inliers, axis=0)
        mad  = np.median(np.abs(inliers - med), axis=0) + 1e-6
        keep = np.all(np.abs(inliers - med) < 2.5 * mad, axis=1)
        new_inliers = inliers[keep]
        if new_inliers.shape[0] < 30 or new_inliers.shape[0] >= inliers.shape[0]:
            inliers = new_inliers if new_inliers.shape[0] >= 30 else inliers
            break
        inliers = new_inliers
    return inliers.mean(axis=0).astype(np.float32), int(inliers.shape[0])


def _E_forward(E: np.ndarray) -> np.ndarray:
    """Unit camera-forward axis in world space, from the extrinsic matrix.

    Matches the convention used in the shift decomposition below
    (row 2 of the rotation block).
    """
    f = np.asarray(E[:3, :3][2, :], dtype=np.float64)
    return f / (np.linalg.norm(f) + 1e-9)


def _front_surface_anchor(points: np.ndarray, fwd_world: np.ndarray,
                          pct: float = 10.0) -> tuple[np.ndarray, int]:
    """Mean of the points nearest the camera along `fwd_world` (front face).

    A robust CENTROID is the wrong statistic for comparing a photo's VGGT
    cloud against a render's: the photo only ever contains the object's
    VISIBLE FRONT SURFACE, while the render's cloud is whatever the placed
    mesh exposes — a different subset of the same object.  Differencing two
    centroids therefore folds the object's own depth extent and its occlusion
    pattern into what is supposed to be a pure position delta, which is why
    office8's depth corrections collapsed to 2-3 cm of noise.

    The near face is the one quantity the segmentation mask faithfully
    preserves in BOTH clouds, so anchor on it: take the points inside the
    lowest `pct` percentile of camera-forward distance and average them.
    A percentile (not the min) keeps it robust to single-pixel flyers.
    """
    if points.shape[0] == 0:
        return np.array([np.nan] * 3, dtype=np.float32), 0
    pts = points.astype(np.float64)
    fwd = np.asarray(fwd_world, dtype=np.float64)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    d = pts @ fwd                                    # signed camera-forward distance
    thresh = np.percentile(d, pct)
    near = pts[d <= thresh]
    if near.shape[0] < 10:                           # too thin a slice — widen
        near = pts[d <= np.percentile(d, min(pct * 3.0, 50.0))]
    if near.shape[0] == 0:
        return np.array([np.nan] * 3, dtype=np.float32), 0
    # MAD reject within the slice so a stray near-field pixel cannot drag it.
    med = np.median(near, axis=0)
    mad = np.median(np.abs(near - med), axis=0) + 1e-6
    keep = np.all(np.abs(near - med) < 3.0 * mad, axis=1)
    if keep.sum() >= 10:
        near = near[keep]
    return near.mean(axis=0).astype(np.float32), int(near.shape[0])


def _fit_sim3(src: np.ndarray, dst: np.ndarray, voxel: float = 0.05
              ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Returns (s, R, t, fitness, rmse) such that  s·R·src + t  ≈ dst."""
    import open3d as o3d
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    src_diag = float(np.linalg.norm(src.max(axis=0) - src.min(axis=0)))
    dst_diag = float(np.linalg.norm(dst.max(axis=0) - dst.min(axis=0)))
    s = dst_diag / src_diag if src_diag > 1e-9 else 1.0
    src_sim = s * (src - src_c) + dst_c
    t0 = dst_c - s * src_c

    src_pcd = o3d.geometry.PointCloud(); src_pcd.points = o3d.utility.Vector3dVector(src_sim.astype(np.float64))
    dst_pcd = o3d.geometry.PointCloud(); dst_pcd.points = o3d.utility.Vector3dVector(dst.astype(np.float64))
    src_ds = src_pcd.voxel_down_sample(voxel)
    dst_ds = dst_pcd.voxel_down_sample(voxel)
    res = o3d.pipelines.registration.registration_icp(
        src_ds, dst_ds, voxel * 4.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    T = np.asarray(res.transformation)
    R_icp, t_icp = T[:3, :3], T[:3, 3]
    R_total = R_icp
    t_total = R_icp @ t0 + t_icp
    return float(s), R_total, t_total, float(res.fitness), float(res.inlier_rmse)


def _apply_sim3(p: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * p) @ R.T + t


# ─── Public API ────────────────────────────────────────────────────────────────

def _sample_walls_metric(walls_obj: Path, n_samples: int = 100_000) -> np.ndarray:
    """Sample surface points (meters) of the empty-room shell."""
    import trimesh
    mesh = trimesh.load(str(walls_obj), force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return np.asarray(pts, dtype=np.float32)


def compute_pseudo_to_metric_polyfit(scene_dir: Path | str,
                                      degree: int = 2,
                                      pixel_step: int = 4,
                                      ) -> "np.ndarray | None":
    """Per-pixel polynomial calibration: metric_depth ≈ poly(pseudo_depth).

    Better than the global scalar when VGGT's near-field depth is non-uniformly
    compressed (which it usually is — near objects predicted shallower than
    far ones).  For each pixel we ray-cast against the room's axis-aligned
    walls (taken from `walls.obj`), get the metric depth at the hit, pair
    with VGGT's pseudo depth at that pixel, and fit a low-order polynomial.

    Returns numpy poly coefficients (highest-degree first, np.polyfit order).
    Apply with `np.polyval(coeffs, pseudo_depth_value)` to convert back to
    metric meters along the camera optical axis.

    Returns None if any required input is missing or the fit doesn't have
    enough valid pairs.
    """
    scene_dir = Path(scene_dir).resolve()
    photo_depth_path = scene_dir / "vggt" / "depth_0.npy"
    cam_path         = scene_dir / "camera_vggt.json"
    walls_path       = scene_dir / "walls.obj"
    if not all(p.exists() for p in (photo_depth_path, cam_path, walls_path)):
        return None

    pseudo_depth = np.load(photo_depth_path)
    cam = json.loads(cam_path.read_text())
    K, E, W, H = _camera_to_KE(cam)
    if pseudo_depth.shape != (H, W):
        return None

    # Load walls.obj vertices to get the room bounding box (axis-aligned).
    try:
        import trimesh as _tm
        mesh = _tm.load(str(walls_path), force="mesh", process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
    except Exception:
        return None
    if len(verts) < 8:
        return None
    x_min, y_min, z_min = verts.min(axis=0)
    x_max, y_max, z_max = verts.max(axis=0)
    bounds = {0: (x_min, x_max), 1: (y_min, y_max), 2: (z_min, z_max)}
    planes = [
        (0, x_min), (0, x_max),
        (1, y_min), (1, y_max),
        (2, z_min), (2, z_max),
    ]

    # Camera basis from look-at spec (matches `_camera_to_KE`).
    pos     = np.array(cam["position_m"], dtype=np.float64)
    look_at = np.array(cam["look_at_m"],  dtype=np.float64)
    up_w    = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    fwd     = look_at - pos; fwd /= np.linalg.norm(fwd)
    right_w = np.cross(fwd, up_w); right_w /= np.linalg.norm(right_w)
    cam_up  = np.cross(right_w, fwd)
    fx_K    = float(K[0, 0])
    cx_K    = float(K[0, 2])
    cy_K    = float(K[1, 2])

    # Vectorised ray-plane intersection on a sub-sampled pixel grid.
    ys, xs = np.mgrid[:H:pixel_step, :W:pixel_step]
    ys = ys.ravel(); xs = xs.ravel()
    pd_pix = pseudo_depth[ys, xs].astype(np.float64)
    valid_pd = (pd_pix > 0) & np.isfinite(pd_pix)
    ys, xs, pd_pix = ys[valid_pd], xs[valid_pd], pd_pix[valid_pd]
    if len(pd_pix) == 0:
        return None
    # World rays for each pixel: ray = x_n*right + y_n*up + fwd, normalised.
    x_n = (xs.astype(np.float64) - cx_K) / fx_K
    y_n = -(ys.astype(np.float64) - cy_K) / fx_K   # image-y down → world-up
    rays = (x_n[:, None] * right_w[None, :]
          + y_n[:, None] * cam_up[None, :]
          + fwd[None, :])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12

    # Intersect each ray with each plane; take smallest positive t with hit
    # inside the room bbox.
    n = rays.shape[0]
    best_cam_z = np.full(n, np.inf, dtype=np.float64)
    for ax, plane_val in planes:
        rd = rays[:, ax]
        denom = np.where(np.abs(rd) > 1e-9, rd, np.nan)
        t = (plane_val - pos[ax]) / denom
        valid = np.isfinite(t) & (t > 0.01)
        if not valid.any():
            continue
        hits = pos[None, :] + t[:, None] * rays
        for ax2 in (0, 1, 2):
            if ax2 == ax:
                continue
            lo, hi = bounds[ax2]
            valid &= (hits[:, ax2] >= lo - 0.02) & (hits[:, ax2] <= hi + 0.02)
        # Camera-Z (depth along optical axis) = (hit - pos) · fwd
        cam_z = ((hits - pos[None, :]) @ fwd).astype(np.float64)
        valid &= (cam_z > 0.01)
        # Update best where this hit is closer
        better = valid & (cam_z < best_cam_z)
        best_cam_z = np.where(better, cam_z, best_cam_z)

    valid_md = np.isfinite(best_cam_z)
    pseudo_vals = pd_pix[valid_md]
    metric_vals = best_cam_z[valid_md]
    if len(pseudo_vals) < 100:
        return None

    # Trim outliers (pseudo with no nearby metric, etc.) by IQR on the residual.
    coeffs0 = np.polyfit(pseudo_vals, metric_vals, degree)
    pred0 = np.polyval(coeffs0, pseudo_vals)
    res0 = metric_vals - pred0
    q1, q3 = np.percentile(res0, [10, 90])
    iqr_keep = (res0 >= q1) & (res0 <= q3)
    if iqr_keep.sum() >= 100:
        pseudo_vals = pseudo_vals[iqr_keep]
        metric_vals = metric_vals[iqr_keep]

    coeffs = np.polyfit(pseudo_vals, metric_vals, degree)
    pred  = np.polyval(coeffs, pseudo_vals)
    ss_res = float(np.sum((metric_vals - pred) ** 2))
    ss_tot = float(np.sum((metric_vals - metric_vals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"[vggt-refine] depth polyfit (deg={degree}, n={len(pseudo_vals)}): "
          f"coeffs={coeffs.round(4).tolist()}  R²={r2:.3f}  "
          f"pseudo∈[{pseudo_vals.min():.2f},{pseudo_vals.max():.2f}] "
          f"→ metric∈[{metric_vals.min():.2f},{metric_vals.max():.2f}]m")
    return coeffs


def compute_pseudo_to_metric_scale(scene_dir: Path | str,
                                    sample: int = 200_000) -> float | None:
    """Estimate the scalar that converts a VGGT-pseudo-world distance into
    real meters: scale = walls_diag / photo_pseudo_diag.

    The photo VGGT cloud (unprojected with the HARMONY camera) and
    `walls.obj` describe the same room at different scales — the bbox-diagonal
    ratio is a robust scale estimate (perspective is identical, only depth
    magnitude differs)."""
    scene_dir = Path(scene_dir).resolve()
    photo_depth_path = scene_dir / "vggt" / "depth_0.npy"
    cam_path         = scene_dir / "camera_vggt.json"
    walls_path       = scene_dir / "walls.obj"
    if not all(p.exists() for p in (photo_depth_path, cam_path, walls_path)):
        return None
    depth = np.load(photo_depth_path)
    cam   = json.loads(cam_path.read_text())
    K, E, W, H = _camera_to_KE(cam)
    if depth.shape != (H, W):
        return None
    photo_pseudo = _unproject_depth(depth, K, E)
    if photo_pseudo.shape[0] > sample:
        photo_pseudo = photo_pseudo[np.random.default_rng(0).choice(
            photo_pseudo.shape[0], sample, replace=False)]
    pseudo_diag = float(np.linalg.norm(photo_pseudo.max(0) - photo_pseudo.min(0)))
    if pseudo_diag <= 1e-6:
        return None
    walls_pts = _sample_walls_metric(walls_path, n_samples=20_000)
    metric_diag = float(np.linalg.norm(walls_pts.max(0) - walls_pts.min(0)))
    s = metric_diag / pseudo_diag
    print(f"[vggt-refine] pseudo→metric scale = {s:.3f} "
          f"(photo pseudo bbox {pseudo_diag:.3f}, walls metric bbox {metric_diag:.3f})")
    return s


def compute_vggt_calibration(scene_dir: Path | str,
                              empty_render: str = "render_vggt.png",
                              cache_subdir: str = "depth_alignment/vggt_empty"
                              ) -> dict[str, Any] | None:
    """Compute Sim(3): photo-VGGT pseudo-world → HARMONY world meters.

    The metric reference is `<scene>/walls.obj` (the bare room shell, in
    meters). Aligning the photo's VGGT pseudo-world cloud to that gives a
    Sim(3) that genuinely converts VGGT-depth scale → real meters.

    A previous version aligned two non-metric clouds against each other,
    yielding a near-identity scale that did NOT convert to meters — that
    produced wildly wrong predictions far from the placed items.

    Returns None if any required input is missing (caller should fall through
    to plain mask_align in that case).
    """
    scene_dir = Path(scene_dir).resolve()
    photo_depth_path = scene_dir / "vggt" / "depth_0.npy"
    cam_path         = scene_dir / "camera_vggt.json"
    walls_path       = scene_dir / "walls.obj"
    for p in (photo_depth_path, cam_path, walls_path):
        if not p.exists():
            print(f"[vggt-refine] missing {p}; skipping VGGT refinement")
            return None

    # Add repo root to sys.path so any inference imports (if needed) work.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    photo_depth = np.load(photo_depth_path)
    cam_json    = json.loads(cam_path.read_text())
    K, E, W, H  = _camera_to_KE(cam_json)
    if photo_depth.shape != (H, W):
        print(f"[vggt-refine] depth {photo_depth.shape} ≠ camera ({H},{W}); skipping")
        return None

    # Photo VGGT depth → pseudo-world (correct shape, wrong scale).
    photo_pseudo = _unproject_depth(photo_depth, K, E)
    rng = np.random.default_rng(0)
    if photo_pseudo.shape[0] > 200_000:
        photo_pseudo = photo_pseudo[rng.choice(photo_pseudo.shape[0], 200_000, replace=False)]

    # Metric reference: surface samples of walls.obj (in meters).
    walls_pts = _sample_walls_metric(walls_path, n_samples=100_000)
    print(f"[vggt-refine] walls.obj reference: {walls_pts.shape[0]} pts, "
          f"bbox extent {tuple((walls_pts.max(0) - walls_pts.min(0)).round(2).tolist())} m")

    s, R, t, fitness, rmse = _fit_sim3(photo_pseudo, walls_pts, voxel=0.05)
    print(f"[vggt-refine] calib: scale={s:.3f}  |t|={np.linalg.norm(t):.3f}m  "
          f"fitness={fitness:.3f}  rmse={rmse:.3f}m")

    # ── Save calibration visualization (aligned point clouds) ────────────────
    try:
        viz_dir = scene_dir / "depth_alignment" / "vggt_refine_viz" / "calibration"
        viz_dir.mkdir(parents=True, exist_ok=True)
        photo_aligned = (s * photo_pseudo) @ R.T + t          # photo VGGT → world meters
        # Combined PLY: red = photo (aligned), gray = walls reference
        red  = np.tile(np.array([220, 30, 30],  dtype=np.uint8), (photo_aligned.shape[0], 1))
        gray = np.tile(np.array([170, 170, 170], dtype=np.uint8), (walls_pts.shape[0], 1))
        _write_ply_simple(viz_dir / "photo_aligned.ply", photo_aligned, red)
        _write_ply_simple(viz_dir / "walls_reference.ply", walls_pts, gray)
        _write_ply_simple(viz_dir / "comparison.ply",
                          np.vstack([photo_aligned, walls_pts]),
                          np.vstack([red, gray]))
        _save_calibration_views(viz_dir / "calibration_views.png",
                                photo_pseudo, photo_aligned, walls_pts,
                                stats={"scale": s, "fitness": fitness, "rmse_m": rmse,
                                       "translation_m": t.tolist()})
        print(f"[vggt-refine] calibration viz → {viz_dir}/")
    except Exception as _ce:
        print(f"[vggt-refine] calibration viz save failed: {_ce}")

    return {
        "scale": s, "R": R, "t": t,
        "fitness": fitness, "rmse": rmse,
        "K": K, "E": E,
        "photo_depth": photo_depth,
    }


def _write_ply_simple(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    n = points.shape[0]
    has_color = colors is not None
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        if has_color:
            arr = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
            arr["xyz"] = points.astype(np.float32); arr["rgb"] = colors.astype(np.uint8)
        else:
            arr = np.empty(n, dtype=[("xyz", "<f4", 3)])
            arr["xyz"] = points.astype(np.float32)
        f.write(arr.tobytes())


def _save_calibration_views(out_path: Path, photo_pseudo: np.ndarray,
                            photo_aligned: np.ndarray, walls_pts: np.ndarray,
                            stats: dict, max_pts: int = 30000) -> None:
    """2×3 grid: top row before-align, bottom row after-align; columns = top/front/side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sub(p):
        return p if p.shape[0] <= max_pts else \
               p[np.random.default_rng(0).choice(p.shape[0], max_pts, replace=False)]

    pp, pa, wp = sub(photo_pseudo), sub(photo_aligned), sub(walls_pts)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    proj = [(0, 2, "top (X-Z)"), (0, 1, "front (X-Y)"), (2, 1, "side (Z-Y)")]
    for col, (i, j, title) in enumerate(proj):
        for row, (cloud, label, colour) in enumerate([
            (pp, "photo VGGT pseudo-world (pre-align)", "red"),
            (pa, "photo VGGT (after Sim3 → meters)",    "red"),
        ]):
            ax = axes[row, col]
            ax.scatter(wp[:, i],     wp[:, j],     s=0.5, c="gray", alpha=0.4, label="walls.obj (m)")
            ax.scatter(cloud[:, i],  cloud[:, j],  s=0.5, c=colour, alpha=0.4, label=label)
            ax.set_title(f"{title} — {'before' if row == 0 else 'after'} align")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(markerscale=8, loc="best", fontsize=8)
    fig.suptitle(f"VGGT calibration — scale={stats['scale']:.3f}  "
                 f"fitness={stats['fitness']:.3f}  rmse={stats['rmse_m']:.3f}m")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def predict_world_position(mask_path: Path | str, calib: dict) -> tuple[np.ndarray, int] | None:
    """Predict world-meter (X,Y,Z) for a furniture mask using calib['photo_depth']."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None
    photo_depth = calib["photo_depth"]
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if mask.shape != photo_depth.shape:
        mask = np.array(Image.open(mask_path).convert("L").resize(
            (photo_depth.shape[1], photo_depth.shape[0]), Image.NEAREST)) > 0
    if int(mask.sum()) < 50:
        return None

    pts = _unproject_depth(photo_depth, calib["K"], calib["E"], pixel_mask=mask)
    if pts.shape[0] == 0:
        return None
    centroid_pseudo, n_inliers = _dominant_centroid(pts)
    centroid_world = _apply_sim3(centroid_pseudo[None, :], calib["scale"], calib["R"], calib["t"])[0]
    return centroid_world.astype(np.float64), n_inliers


def _save_object_viz(viz_dir: Path, p: dict, entry: dict, mask_path: Path,
                     pred_pos: np.ndarray, pre_pos: np.ndarray, post_pos: np.ndarray,
                     n_inliers: int, clipped: bool) -> None:
    """Save a per-object diagnostic PNG: masked crop + top-down plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir.mkdir(parents=True, exist_ok=True)
    idx, ftype = p.get("index", -1), p.get("type", entry.get("type", "?"))
    out_path = viz_dir / f"vggt_refine_{idx:02d}_{ftype}.png"

    # Try to load reference photo from <scene>/vggt/camera.json (which records
    # the input image path used by the VGGT inference run).
    scene_dir = mask_path.parents[2]   # <scene>/furniture/segmented/<file>
    photo_path: Path | None = None
    vggt_cam_json = scene_dir / "vggt" / "camera.json"
    if vggt_cam_json.exists():
        try:
            data = json.loads(vggt_cam_json.read_text())
            entries = data if isinstance(data, list) else [data]
            ipath = entries[0].get("image") if entries else None
            if isinstance(ipath, str):
                cand = Path(ipath)
                if not cand.is_absolute():
                    cand = Path(__file__).resolve().parents[2] / cand
                if cand.exists():
                    photo_path = cand
        except Exception:
            photo_path = None
    photo = Image.open(photo_path).convert("RGB") if photo_path else None
    mask = Image.open(mask_path).convert("L")
    if photo is not None and mask.size != photo.size:
        mask = mask.resize(photo.size, Image.NEAREST)
    mask_arr = np.array(mask) > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: photo with mask outline + bbox.
    ax = axes[0]
    if photo is not None:
        ax.imshow(np.array(photo))
        # crop bbox around the mask for context
        ys, xs = np.where(mask_arr)
        if len(xs) > 0:
            x1, x2 = max(int(xs.min()) - 50, 0), min(int(xs.max()) + 50, photo.width)
            y1, y2 = max(int(ys.min()) - 50, 0), min(int(ys.max()) + 50, photo.height)
            ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
            # mask overlay (semi-transparent red)
            overlay = np.zeros((*mask_arr.shape, 4), dtype=np.float32)
            overlay[mask_arr] = [1.0, 0.2, 0.2, 0.35]
            ax.imshow(overlay)
    else:
        ax.imshow(mask_arr.astype(np.uint8) * 255, cmap="gray")
    ax.set_title(f"idx={idx}  type={ftype}\nmask file: {mask_path.name}")
    ax.set_xticks([]); ax.set_yticks([])

    # Right: top-down (X,Z) plot — placed vs predicted vs post-refine.
    ax = axes[1]
    pts = np.array([pre_pos, pred_pos, post_pos])
    ax.scatter([pre_pos[0]],  [pre_pos[2]],  c="black", s=120, marker="s", label="placed front face")
    ax.scatter([pred_pos[0]], [pred_pos[2]], c="red",   s=100, marker="o", label="VGGT prediction (front face)")
    ax.scatter([post_pos[0]], [post_pos[2]], c="green", s=100, marker="^", label="post-refine front face")
    ax.plot([pre_pos[0], pred_pos[0]], [pre_pos[2], pred_pos[2]], "k--", alpha=0.4)
    ax.plot([pre_pos[0], post_pos[0]], [pre_pos[2], post_pos[2]], "g-",  alpha=0.6, lw=2)

    delta = post_pos - pre_pos
    pred_delta = pred_pos - pre_pos
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    ax.set_title(
        f"top-down\n"
        f"pre→pred Δxz={float(np.linalg.norm(pred_delta[[0,2]])):.3f}m, "
        f"pre→post Δxz={float(np.linalg.norm(delta[[0,2]])):.3f}m  "
        f"{'CLIPPED' if clipped else ''}\n"
        f"inliers={n_inliers}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _center_offset_from_visible_face(p: dict, calib: dict) -> np.ndarray:
    """Mask covers only the visible (camera-facing) face of furniture, so the
    masked-pixel centroid sits on that face — biased toward the camera. Push
    it back along world-forward by the OBB's half-extent along that axis,
    which is the true offset from "front face midpoint" to "bbox center"
    regardless of how the object is rotated.

        half_extent_along_fwd = Σᵢ |fwd · object_axis_i| · half_extent_i

    For a sofa parallel to the back wall (long axis ≠ camera depth axis) this
    is much larger than its local depth_m / 2, which is why the previous
    half-depth heuristic kept the prediction too "fronter".
    """
    fwd_world  = calib["E"][:3, :3][2, :].astype(np.float64)
    fwd_world /= np.linalg.norm(fwd_world) + 1e-9

    sm = p.get("size_m") or {}
    half_local = np.array([
        float(sm.get("width_m",  0.5)) / 2.0,
        float(sm.get("height_m", 0.5)) / 2.0,
        float(sm.get("depth_m",  0.5)) / 2.0,
    ], dtype=np.float64)

    R_obj = np.array(p.get("rotation_3x3") or np.eye(3).tolist(), dtype=np.float64)
    # Columns of R_obj are object axes expressed in world coords.
    proj = np.abs(R_obj.T @ fwd_world)  # |fwd · axis_i|, shape (3,)
    half_along_fwd = float((proj * half_local).sum())
    half_along_fwd = max(0.05, min(half_along_fwd, 0.80))
    return (half_along_fwd * fwd_world).astype(np.float64)


def refine_placement(p: dict, entry: dict, scene_dir: Path | str, calib: dict | None,
                     blend: float = 0.5, max_shift_m: float = 0.6,
                     respect_y: bool = True,
                     visible_face_offset: bool = False,
                     lateral_weight: float = 1.0,
                     depth_weight:   float = 0.0,
                     save_viz: bool = True,
                     all_placements: list[dict] | None = None) -> dict:
    """Refine `p["position_m"]` using the VGGT-predicted world position from
    the entry's segmentation mask.

    Args:
        p:           The placement dict (will be modified in place AND returned).
        entry:       The matching entry from segment_results (provides `mask_file`).
        scene_dir:   The scene's outputs/<scene>/ root.
        calib:       Output of compute_vggt_calibration. None disables refinement.
        blend:       0.0 = ignore VGGT, 1.0 = snap to VGGT, 0.5 = average.
        max_shift_m: Cap on how far VGGT may pull the placement (XZ only).
                     Larger drifts are clamped (suggests bad mask or VGGT noise).
        respect_y:   If True, keep p["position_m"][1] (height) untouched
                     (silhouette mask_align already gets Y from floor sit).

    Stores diagnostic fields under `p["_vggt_refine"]`.
    """
    if calib is None:
        return p

    mask_file = entry.get("mask_file", "")
    if not mask_file:
        return p
    scene_dir = Path(scene_dir).resolve()
    mask_path = scene_dir / "furniture" / "segmented" / mask_file

    pred = predict_world_position(mask_path, calib)
    if pred is None:
        p.setdefault("_vggt_refine", {})["status"] = "no_prediction"
        return p
    pred_raw, n_inliers = pred  # raw centroid (visible-surface average)

    # Decompose the suggested shift into a camera-forward (depth) component
    # and a camera-right (lateral) component, then weight them separately.
    # Reasoning: VGGT's depth is unreliable (mask sits on the visible front
    # face, not the bbox center; VGGT also has scale ambiguity). Lateral
    # position is robust — it's just "where the object is in the image".
    cur_pos = np.array(p.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64)

    fwd_world   = calib["E"][:3, :3][2, :].astype(np.float64)
    right_world = calib["E"][:3, :3][0, :].astype(np.float64)
    fwd_world  /= np.linalg.norm(fwd_world)   + 1e-9
    right_world /= np.linalg.norm(right_world) + 1e-9
    # Project both to XZ horizontal plane (we only edit X,Z).
    fwd_xz   = np.array([fwd_world[0],   fwd_world[2]],   dtype=np.float64)
    right_xz = np.array([right_world[0], right_world[2]], dtype=np.float64)
    fwd_xz   /= np.linalg.norm(fwd_xz)   + 1e-9
    right_xz /= np.linalg.norm(right_xz) + 1e-9

    raw_shift_xz = (pred_raw - cur_pos)[[0, 2]]      # what VGGT would move us by
    shift_fwd    = float(raw_shift_xz @ fwd_xz)      # depth component
    shift_lat    = float(raw_shift_xz @ right_xz)    # lateral component

    # ── evidence-weighted blend ──────────────────────────────────────────────
    # `blend` was a flat constant, so a prediction backed by 1,935 surviving
    # points was trusted exactly as much as one backed by 168,566.  Both of
    # those are real: living_room8's floor lamp (1,935) and pexels_2343465's
    # armchair (168,566) each had their VGGT correction halved by the same 0.5.
    # That is backwards — the whole reason to discount VGGT is uncertainty, and
    # the inlier count is a direct measure of how much surface the mask actually
    # observed.  Scale the blend by support, log-spaced because inlier counts
    # span two orders of magnitude across a single scene.
    _n_lo   = float(os.environ.get("SCENEWEAVE_VGGT_N_LO",   "2000"))
    _n_hi   = float(os.environ.get("SCENEWEAVE_VGGT_N_HI",   "50000"))
    _b_lo   = float(os.environ.get("SCENEWEAVE_VGGT_BLEND_LO", "0.30"))
    _b_hi   = float(os.environ.get("SCENEWEAVE_VGGT_BLEND_HI", "0.80"))
    if os.environ.get("SCENEWEAVE_VGGT_FLAT_BLEND") == "1":
        blend_eff, _conf = float(blend), None      # old behaviour, for A/B
    else:
        _conf = (math.log10(max(float(n_inliers), 1.0) / _n_lo)
                 / math.log10(_n_hi / _n_lo))
        _conf = min(1.0, max(0.0, _conf))
        # Anchor on the caller's blend so an explicit override still scales:
        # at the default blend=0.5 this spans 0.30 … 0.80.
        blend_eff = (_b_lo + (_b_hi - _b_lo) * _conf) * (float(blend) / 0.5)
        blend_eff = min(1.0, max(0.0, blend_eff))

    # Apply per-axis weights (default: keep all lateral, drop all depth).
    weighted_shift = (
        depth_weight   * blend_eff * shift_fwd * fwd_xz +
        lateral_weight * blend_eff * shift_lat * right_xz
    )

    # Cap total magnitude.
    shift_mag = float(np.linalg.norm(weighted_shift))
    clipped   = shift_mag > max_shift_m
    if clipped:
        weighted_shift = weighted_shift * (max_shift_m / shift_mag)

    blended_center = cur_pos.copy()
    blended_center[0] += weighted_shift[0]
    blended_center[2] += weighted_shift[1]
    if not respect_y:
        blended_center[1] = pred_raw[1]

    # Front-face positions (for visualization parity with the mask centroid).
    if visible_face_offset:
        front_offset = _center_offset_from_visible_face(p, calib)
    else:
        front_offset = np.zeros(3, dtype=np.float64)
    pred_front    = pred_raw - front_offset
    placed_front  = cur_pos - front_offset
    blended_front = blended_center - front_offset

    p.setdefault("_vggt_refine", {}).update({
        "status": "applied",
        "raw_predicted_m":         pred_raw.tolist(),
        "shift_fwd_raw_m":         shift_fwd,         # signed: +fwd = away from camera
        "shift_lateral_raw_m":     shift_lat,         # signed: +right
        "shift_fwd_applied_m":     float(weighted_shift @ np.eye(2)[0] * fwd_xz[0] +
                                         weighted_shift @ np.eye(2)[1] * fwd_xz[1]),
        "shift_lateral_applied_m": float(weighted_shift @ np.eye(2)[0] * right_xz[0] +
                                         weighted_shift @ np.eye(2)[1] * right_xz[1]),
        "lateral_weight": float(lateral_weight),
        "depth_weight":   float(depth_weight),
        # What the evidence weighting actually decided, so a placement can be
        # explained after the fact rather than just observed.
        "blend_nominal":  float(blend),
        "blend_effective": round(float(blend_eff), 4),
        "support_conf":   (None if _conf is None else round(float(_conf), 4)),
        "front_offset_m":         front_offset.tolist(),
        "placed_front_m":         placed_front.tolist(),
        "predicted_front_m":      pred_raw.tolist(),
        "blended_front_m":        blended_front.tolist(),
        "pre_refine_position_m":  cur_pos.tolist(),
        "post_refine_position_m": blended_center.tolist(),
        "shift_xz_m":             float(np.linalg.norm(blended_center[[0, 2]] - cur_pos[[0, 2]])),
        "n_inliers":              int(n_inliers),
        "blend":                  float(blend),
        "clipped":                clipped,
    })
    p["position_m"] = blended_center.tolist()

    if save_viz and not _ABL_NO_VIZ:
        try:
            viz_dir = Path(scene_dir) / "depth_alignment" / "vggt_refine_viz"
            _save_object_viz(viz_dir, p, entry, mask_path,
                             pred_front, placed_front, blended_front,
                             n_inliers=n_inliers, clipped=shift_mag > max_shift_m)
        except Exception as _ve:
            print(f"  [vggt_refine] viz save failed for idx={p.get('index')}: {_ve}")
        # Also refresh the running scene-wide summary so it grows incrementally
        # as each object is refined (not only at the end of the placement loop).
        try:
            running = list(all_placements) if all_placements is not None else []
            if p not in running:
                running = running + [p]
            save_summary_plot(scene_dir, running, out_name="_summary.png")
        except Exception as _se:
            print(f"  [vggt_refine] running summary save failed: {_se}")
    return p


def _main_calibration_only() -> None:
    """CLI: just (re)compute the VGGT-to-world calibration for a scene and
    write the calibration visualization. Useful for iterating on the
    calibration step without rerunning furniture placement."""
    import argparse
    ap = argparse.ArgumentParser(description=_main_calibration_only.__doc__)
    ap.add_argument("--output-dir", required=True,
                    help="Scene dir (e.g. outputs/office8_animation)")
    args = ap.parse_args()
    calib = compute_vggt_calibration(args.output_dir)
    if calib is None:
        print("[vggt-refine] calibration failed; check missing inputs above")
        return
    print(f"[vggt-refine] done — view "
          f"{Path(args.output_dir)/'depth_alignment/vggt_refine_viz/calibration/'}")


def render_diff_refine_placement(
    p: dict, entry: dict, scene_dir: Path | str,
    ref_depth: np.ndarray, K: np.ndarray, E: np.ndarray,
    render_callback,                 # callable(placements, out_path) -> None
    all_placements: list[dict],
    blend: float = 0.5, max_shift_m: float = 0.6,
    lateral_weight: float = 1.0, depth_weight: float = 0.0,
    pseudo_to_metric_scale: float = 1.0,
    pseudo_to_metric_polyfit: "np.ndarray | None" = None,
    save_viz: bool = True,
) -> dict:
    """Per-iteration VGGT refinement.

    Pipeline:
      1. Render the current scene with `p` placed (via `render_callback`).
      2. Run VGGT on that render → render_depth (metric-ish, in VGGT's own scale).
      3. For the same per-furniture mask, take the median centroid of the
         masked pixels in BOTH `ref_depth` (VGGT on the photo, cached) and
         `render_depth` (VGGT on the just-made render).
      4. Both clouds are unprojected with the SAME camera and BOTH have the
         visible-face vs bbox-center bias, so it CANCELS in the difference.
      5. Shift = ref_centroid − render_centroid. Apply with lateral/depth
         weighting, capped by max_shift_m.
    """
    if render_callback is None:
        return p
    mask_file = entry.get("mask_file", "")
    if not mask_file:
        return p
    scene_dir = Path(scene_dir).resolve()
    mask_path = scene_dir / "furniture" / "segmented" / mask_file
    if not mask_path.exists():
        return p

    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if mask.shape != ref_depth.shape:
        mask = np.array(Image.open(mask_path).convert("L").resize(
            (ref_depth.shape[1], ref_depth.shape[0]), Image.NEAREST)) > 0
    if int(mask.sum()) < 50:
        p.setdefault("_vggt_refine", {})["status"] = "mask_too_small"
        return p

    idx = int(p.get("index", -1))
    work_dir = scene_dir / "depth_alignment" / "vggt_refine_viz" / "render_diff"
    work_dir.mkdir(parents=True, exist_ok=True)
    render_path = work_dir / f"render_idx{idx:02d}.png"
    vggt_cache  = work_dir / f"vggt_idx{idx:02d}"

    # 1. Render current placements + p.
    try:
        render_callback(list(all_placements) + [p], render_path)
    except Exception as _re:
        p.setdefault("_vggt_refine", {})["status"] = f"render_failed: {_re}"
        return p
    if not render_path.exists():
        p.setdefault("_vggt_refine", {})["status"] = "render_missing"
        return p

    # 2. Run VGGT (cache per-idx so repeated calls reuse the result).
    if not (vggt_cache / "depth_0.npy").exists():
        try:
            from floorplan.vggt_estimates.inference import run as vggt_run
            vggt_run([str(render_path)], out_dir=str(vggt_cache))
        except Exception as _ve:
            p.setdefault("_vggt_refine", {})["status"] = f"vggt_failed: {_ve}"
            return p
    render_depth = np.load(vggt_cache / "depth_0.npy")
    if render_depth.shape != ref_depth.shape:
        # Resize render depth to match ref depth shape (nearest, since it's depth).
        from PIL import Image as _PI
        rd = _PI.fromarray(render_depth.astype(np.float32))
        rd = rd.resize((ref_depth.shape[1], ref_depth.shape[0]), _PI.NEAREST)
        render_depth = np.array(rd, dtype=np.float32)

    # 3. Extract masked centroids in both clouds.
    # Per-pixel depth conversion: if a polynomial fit is available, convert
    # both depth maps from VGGT-pseudo to metric BEFORE unprojection.  This
    # handles VGGT's non-uniform near-field compression correctly (near
    # objects predicted shallower than far ones).  Without this, a single
    # global scalar would systematically pull near objects toward the camera
    # and push far ones away.
    if pseudo_to_metric_polyfit is not None:
        ref_depth_m    = np.polyval(pseudo_to_metric_polyfit, ref_depth)
        render_depth_m = np.polyval(pseudo_to_metric_polyfit, render_depth)
        ref_pts    = _unproject_depth(ref_depth_m,    K, E, pixel_mask=mask)
        render_pts = _unproject_depth(render_depth_m, K, E, pixel_mask=mask)
    else:
        ref_pts    = _unproject_depth(ref_depth,    K, E, pixel_mask=mask)
        render_pts = _unproject_depth(render_depth, K, E, pixel_mask=mask)
    if ref_pts.shape[0] < 50 or render_pts.shape[0] < 50:
        p.setdefault("_vggt_refine", {})["status"] = "too_few_masked_points"
        return p
    ref_c, n_ref     = _dominant_centroid(ref_pts)
    render_c, n_rnd  = _dominant_centroid(render_pts)
    raw_shift_pseudo = (ref_c - render_c).astype(np.float64)

    # ── Front-surface depth anchor ────────────────────────────────────────
    # Keep the CENTROID delta for the lateral component (stable, and the
    # silhouette mask_align owns lateral anyway), but take the camera-forward
    # component from the near face of each cloud instead.  See
    # _front_surface_anchor: the photo cloud holds only the visible front,
    # the render cloud holds the placed mesh's exposed surface, so centroid
    # differencing compares two different subsets of the object and the depth
    # signal drowns in the object's own thickness.
    _fwd_anchor = _E_forward(E)
    if os.environ.get("SCENEWEAVE_VGGT_FRONT_ANCHOR", "1") == "1":
        _pct = float(os.environ.get("SCENEWEAVE_VGGT_FRONT_PCT", "10"))
        ref_f,  n_ref_f  = _front_surface_anchor(ref_pts,    _fwd_anchor, _pct)
        rnd_f,  n_rnd_f  = _front_surface_anchor(render_pts, _fwd_anchor, _pct)
        if n_ref_f >= 10 and n_rnd_f >= 10 and np.all(np.isfinite(ref_f)) \
                and np.all(np.isfinite(rnd_f)):
            _delta_front = (ref_f - rnd_f).astype(np.float64)
            _fwd_c = float(raw_shift_pseudo @ _fwd_anchor)
            _fwd_f = float(_delta_front     @ _fwd_anchor)
            # swap ONLY the forward component
            raw_shift_pseudo = (raw_shift_pseudo
                                - _fwd_c * _fwd_anchor
                                + _fwd_f * _fwd_anchor)
            print(f"  [vggt_refine] front-anchor p{_pct:.0f}: depth delta "
                  f"{_fwd_c:+.3f} → {_fwd_f:+.3f} (pseudo units, "
                  f"n_ref={n_ref_f} n_render={n_rnd_f})")
    if pseudo_to_metric_polyfit is not None:
        # Already in metric thanks to per-pixel poly conversion above.
        raw_shift = raw_shift_pseudo
    else:
        # Fall-back: global scalar (uniform depth scaling).  Less accurate
        # when VGGT's depth distortion is non-uniform.
        raw_shift = raw_shift_pseudo * float(pseudo_to_metric_scale)

    # 4. Decompose into camera-right (lateral) + camera-forward (depth).
    fwd_world   = E[:3, :3][2, :].astype(np.float64); fwd_world  /= np.linalg.norm(fwd_world)   + 1e-9
    right_world = E[:3, :3][0, :].astype(np.float64); right_world /= np.linalg.norm(right_world) + 1e-9
    fwd_xz   = np.array([fwd_world[0],   fwd_world[2]],   dtype=np.float64)
    right_xz = np.array([right_world[0], right_world[2]], dtype=np.float64)
    fwd_xz   /= np.linalg.norm(fwd_xz)   + 1e-9
    right_xz /= np.linalg.norm(right_xz) + 1e-9

    raw_xz = raw_shift[[0, 2]]
    shift_fwd = float(raw_xz @ fwd_xz)
    shift_lat = float(raw_xz @ right_xz)
    weighted = depth_weight * blend * shift_fwd * fwd_xz \
             + lateral_weight * blend * shift_lat * right_xz
    mag = float(np.linalg.norm(weighted))
    clipped = mag > max_shift_m
    if clipped:
        weighted = weighted * (max_shift_m / mag)

    cur_pos = np.array(p.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    new_pos = cur_pos.copy()
    new_pos[0] += weighted[0]
    new_pos[2] += weighted[1]

    # Hard cap on cumulative drift from the silhouette anchor: VGGT predictions
    # are noisy and can pull objects far across the room. The mask-aligned
    # position is the trusted anchor — abandon the shift if it would put the
    # object more than _MAX_MASK_DEVIATION m from where mask_align placed it.
    _MAX_MASK_DEVIATION = 0.40
    _origin = p.get("_mask_origin_xz")
    deviation_clipped = False
    if _origin is not None:
        _drift = float(np.sqrt((new_pos[0] - _origin[0])**2
                               + (new_pos[2] - _origin[1])**2))
        if _drift > _MAX_MASK_DEVIATION:
            # Scale weighted vector down so the new position lands exactly on
            # the deviation circle around the anchor.
            from_anchor = (cur_pos[[0, 2]] + weighted) - np.array(_origin, dtype=np.float64)
            from_anchor_norm = float(np.linalg.norm(from_anchor))
            if from_anchor_norm > 1e-6:
                from_anchor = from_anchor * (_MAX_MASK_DEVIATION / from_anchor_norm)
                clamped_xz = np.array(_origin, dtype=np.float64) + from_anchor
                new_pos[0], new_pos[2] = float(clamped_xz[0]), float(clamped_xz[1])
                weighted = clamped_xz - cur_pos[[0, 2]]
                deviation_clipped = True

    p.setdefault("_vggt_refine", {}).update({
        "status": "applied",
        "method": "render_diff",
        "mask_deviation_clipped": deviation_clipped,
        "render_path":           str(render_path),
        "ref_centroid_m":        ref_c.tolist(),
        "render_centroid_m":     render_c.tolist(),
        "raw_shift_m":           raw_shift.tolist(),
        "shift_fwd_raw_m":       shift_fwd,
        "shift_lateral_raw_m":   shift_lat,
        "lateral_weight":        float(lateral_weight),
        "depth_weight":          float(depth_weight),
        "blend":                 float(blend),
        "pre_refine_position_m":  cur_pos.tolist(),
        "post_refine_position_m": new_pos.tolist(),
        "shift_xz_m":            float(np.linalg.norm(new_pos[[0, 2]] - cur_pos[[0, 2]])),
        "n_inliers_ref":         int(n_ref),
        "n_inliers_render":      int(n_rnd),
        "clipped":               clipped,
        # For visualization parity with the global-calib path:
        "placed_front_m":        cur_pos.tolist(),
        "predicted_front_m":     (cur_pos + raw_shift).tolist(),
        "blended_front_m":       new_pos.tolist(),
    })
    p["position_m"] = new_pos.tolist()

    if save_viz and not _ABL_NO_VIZ:
        try:
            viz_dir = scene_dir / "depth_alignment" / "vggt_refine_viz"
            _save_object_viz(viz_dir, p, entry, mask_path,
                             pred_pos=(cur_pos + raw_shift),
                             pre_pos=cur_pos, post_pos=new_pos,
                             n_inliers=int(n_rnd), clipped=clipped)
        except Exception as _ve:
            print(f"  [vggt_refine] viz save failed for idx={idx}: {_ve}")
        try:
            running = list(all_placements) + ([p] if p not in all_placements else [])
            save_summary_plot(scene_dir, running)
        except Exception as _se:
            print(f"  [vggt_refine] summary save failed: {_se}")
    return p


def save_summary_plot(scene_dir: Path | str, placements: list[dict],
                      out_name: str = "_summary.png") -> Path | None:
    """One top-down plot showing placed front face / VGGT prediction / post-refine
    for every refined item in the scene, with the room walls drawn for context.
    Returns the saved path, or None if nothing was refined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(p, p.get("_vggt_refine"))
            for p in placements
            if isinstance(p.get("_vggt_refine"), dict)
            and p["_vggt_refine"].get("status") == "applied"]
    if not rows:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Draw room walls outline (XZ projection of walls.obj) for context.
    walls_path = Path(scene_dir) / "walls.obj"
    if walls_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(str(walls_path), force="mesh", process=False)
            mn, mx = mesh.bounds[0], mesh.bounds[1]
            ax.plot([mn[0], mx[0], mx[0], mn[0], mn[0]],
                    [mn[2], mn[2], mx[2], mx[2], mn[2]],
                    color="gray", lw=1.5, alpha=0.6, label="walls.obj footprint")
        except Exception:
            pass

    for p, vrf in rows:
        idx, ftype = p.get("index", -1), p.get("type", "?")
        # All three points live on the visible front face (same OBB face),
        # so they're directly comparable.
        pre  = np.array(vrf.get("placed_front_m",    vrf["pre_refine_position_m"]))
        pred = np.array(vrf.get("predicted_front_m", vrf.get("predicted_world_m")))
        post = np.array(vrf.get("blended_front_m",   vrf["post_refine_position_m"]))
        ax.scatter([pre[0]],  [pre[2]],  c="black", s=80, marker="s")
        ax.scatter([pred[0]], [pred[2]], c="red",   s=60, marker="o")
        ax.scatter([post[0]], [post[2]], c="green", s=60, marker="^")
        ax.plot([pre[0], pred[0]], [pre[2], pred[2]], "k--", alpha=0.3, lw=1)
        ax.plot([pre[0], post[0]], [pre[2], post[2]], "g-",  alpha=0.6, lw=2)
        ax.annotate(f"{idx}:{ftype[:8]}", (post[0], post[2]),
                    fontsize=8, alpha=0.75, xytext=(4, 4), textcoords="offset points")

    # Legend proxies
    ax.scatter([], [], c="black", s=80, marker="s", label="placed front face")
    ax.scatter([], [], c="red",   s=60, marker="o", label="VGGT prediction (front face)")
    ax.scatter([], [], c="green", s=60, marker="^", label="post-refine front face")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    ax.set_title(f"VGGT refine summary (front-face XZ) — {len(rows)} object(s)")
    out = Path(scene_dir) / "depth_alignment" / "vggt_refine_viz" / "_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    _main_calibration_only()
