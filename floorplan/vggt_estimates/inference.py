"""
floorplan/vggt_estimates/inference.py
---------------------------
VGGT wrapper that lives inside the HARMONY package.
The vggt repo (harmony/vggt/) is treated as a read-only dependency;
nothing inside it is modified.

Outputs per frame N (all at original image resolution):
  camera.json        — extrinsics (3×4) and intrinsics (3×3) in original px coords
  depth_<N>.npy      — metric depth map (H_orig, W_orig) float32
  depth_<N>.png      — false-colour depth visualisation
  normal_<N>.npy     — world-space surface normals (H_orig, W_orig, 3) float32
  normal_<N>.png     — normal map visualisation
  points.ply         — coloured point cloud

Usage
-----
    # as a module
    from floorplan.vggt_estimates.inference import run
    run(["data/indoor_images/office5.jpg"], out_dir="outputs/vggt_office5")

    # CLI (run from the repo root)
    python -m floorplan.vggt_estimates.inference \\
        --images data/indoor_images/office5.jpg \\
        --out    outputs/vggt_office5
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── inject vggt repo onto sys.path without modifying it ──────────────────────
_VGGT_REPO = Path(__file__).parent.parent.parent / "vggt"
if _VGGT_REPO.exists() and str(_VGGT_REPO) not in sys.path:
    sys.path.insert(0, str(_VGGT_REPO))

from vggt.models.vggt import VGGT                           # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images   # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

_VGGT_TARGET = 518   # fixed input size used by load_and_preprocess_images


# ── helpers ───────────────────────────────────────────────────────────────────

def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalise depth → uint8 RGB using a blue→red colourmap."""
    d = depth.copy().astype(np.float32)
    valid = d > 0
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(d * 2 - 0.5, 0, 1)
    g = np.clip(1 - np.abs(d * 2 - 1), 0, 1)
    b = np.clip(0.5 - d * 2 + 1, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _normals_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Surface normals in camera space via finite differences. Returns (H,W,3) float32."""
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    pts = np.stack([(u - cx) / fx * depth,
                    (v - cy) / fy * depth,
                    depth], axis=-1)
    dxdu = np.diff(pts, axis=1, append=pts[:, -1:, :])
    dydv = np.diff(pts, axis=0, append=pts[-1:, :, :])
    n = np.cross(dxdu, dydv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return n.astype(np.float32)


def _normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    return ((normals * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray,
               conf: np.ndarray | None = None, conf_pct: float = 10.0) -> None:
    if conf is not None:
        mask = conf >= np.percentile(conf, conf_pct)
        points, colors = points[mask], colors[mask]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for p, c in zip(points.astype(np.float32), colors.astype(np.uint8)):
            f.write(struct.pack("<fff", *p))
            f.write(bytes(c[:3]))


def _preprocess_info(image_paths: list[str]) -> list[dict]:
    """
    Return per-image preprocessing metadata that mirrors what
    load_and_preprocess_images(mode="crop") does, so we can invert
    the crop+resize to map outputs back to original pixel coordinates.
    """
    info = []
    for p in image_paths:
        W_o, H_o = Image.open(p).size          # PIL gives (width, height)
        scale      = _VGGT_TARGET / W_o         # AR-preserving resize factor
        new_h      = round(H_o * scale / 14) * 14  # height after AR-resize
        crop_y     = max(0, (new_h - _VGGT_TARGET) // 2)  # centre-crop offset
        info.append(dict(H_o=H_o, W_o=W_o, scale=scale,
                         new_h=new_h, crop_y=crop_y))
    return info


def _upsample_to_orig(arr_proc: np.ndarray, *,
                      H_proc: int, new_h: int, new_w: int,
                      crop_y: int, H_o: int, W_o: int,
                      is_depth: bool) -> np.ndarray:
    """
    Map a processed-space array (H_proc × new_w [× C]) back to (H_o × W_o [× C]).

    The processed image is a centre-crop of the AR-resized image:
      AR-resized size : new_h × new_w
      crop window     : rows [crop_y, crop_y + H_proc)

    Steps
    -----
    1. Allocate zero canvas at (new_h × new_w).
    2. Paste proc content at row offset crop_y.
    3. Edge-extend uncovered top/bottom rows with the nearest valid row.
    4. Resize canvas → (H_o × W_o).
    """
    resample = Image.NEAREST if is_depth else Image.BILINEAR
    squeeze  = arr_proc.ndim == 2
    if squeeze:
        arr_proc = arr_proc[:, :, None]
    C = arr_proc.shape[2]

    out_channels = []
    row_end = crop_y + H_proc
    for c in range(C):
        ch = arr_proc[:, :, c]
        canvas = np.zeros((new_h, new_w), dtype=ch.dtype)
        canvas[crop_y:row_end, :] = ch
        if crop_y > 0:
            canvas[:crop_y, :] = canvas[crop_y, :]
        if row_end < new_h:
            canvas[row_end:, :] = canvas[row_end - 1, :]
        out_channels.append(
            np.array(Image.fromarray(canvas).resize((W_o, H_o), resample))
        )

    result = np.stack(out_channels, axis=-1)
    return result[:, :, 0] if squeeze else result


def _scale_intrinsics(K_proc: np.ndarray, scale: float, crop_y: float) -> np.ndarray:
    """
    Convert a 3×3 intrinsic matrix from processed (518 px) space to original
    image pixel coordinates.

      fx_orig = fx_proc / scale
      cx_orig = cx_proc / scale          (no horizontal crop)
      cy_orig = (cy_proc + crop_y) / scale
    """
    K = K_proc.copy()
    inv = 1.0 / scale
    K[0, 0] *= inv
    K[1, 1] *= inv
    K[0, 2] *= inv
    K[1, 2]  = K[1, 2] * inv + crop_y / scale
    return K


# ── main inference ────────────────────────────────────────────────────────────

def run(image_paths: list[str], out_dir: str = "outputs/vggt",
        conf_pct: float = 10.0, device: str | None = None) -> dict:
    """
    Run VGGT on *image_paths* and write all outputs to *out_dir*.

    All saved maps (depth, normals) are at original input resolution.
    Intrinsics in camera.json are in original pixel coordinates.

    Args:
        image_paths: Input image paths (1 = monocular, >1 = multi-view).
        out_dir:     Destination directory (created if absent).
        conf_pct:    Filter lowest N% confidence points from the PLY.
        device:      "cuda" / "cpu" / None (auto-detect).

    Returns:
        dict with keys: camera, depth, normals, world_points, extrinsics, intrinsics
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vggt] device: {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("[vggt] Loading facebook/VGGT-1B …")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=True).to(device).eval()

    # ── preprocess ────────────────────────────────────────────────────────────
    print(f"[vggt] Preprocessing {len(image_paths)} image(s) …")
    prep_info = _preprocess_info(image_paths)
    images = load_and_preprocess_images(image_paths).to(device, dtype=dtype)
    H_proc, W_proc = images.shape[-2], images.shape[-1]

    # ── inference ─────────────────────────────────────────────────────────────
    print("[vggt] Inference …")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        pred = model(images)

    # pose_encoding_to_extri_intri expects (B, S, 9); [0] pulls batch dim after
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pred["pose_enc"], (H_proc, W_proc)
    )
    extrinsics = extrinsics[0].cpu().numpy()   # (S, 3, 4)
    intrinsics = intrinsics[0].cpu().numpy()   # (S, 3, 3)  — in proc px coords

    depth_all  = pred["depth"][0].cpu().numpy()             # (S, H, W, 1)
    world_pts  = pred["world_points"][0].cpu().numpy()      # (S, H, W, 3)
    world_conf = pred["world_points_conf"][0].cpu().numpy() # (S, H, W)
    images_np  = images.cpu().to(torch.float32).numpy()    # (S, 3, H, W)
    S = extrinsics.shape[0]

    # ── camera.json ───────────────────────────────────────────────────────────
    cam_data = []
    for i in range(S):
        info = prep_info[i] if i < len(prep_info) else {}
        H_o  = info.get("H_o", H_proc)
        W_o  = info.get("W_o", W_proc)
        K_orig = _scale_intrinsics(
            intrinsics[i],
            scale=info.get("scale", 1.0),
            crop_y=info.get("crop_y", 0),
        ) if info else intrinsics[i]
        cam_data.append({
            "frame":              i,
            "image":              str(image_paths[i]) if i < len(image_paths) else None,
            "image_size_hw_orig": [H_o, W_o],
            "image_size_hw_proc": [H_proc, W_proc],
            "extrinsic_3x4":      extrinsics[i].tolist(),
            "intrinsic_3x3":      K_orig.tolist(),  # in original px coords
        })
    (out / "camera.json").write_text(json.dumps(cam_data, indent=2))
    print(f"[vggt] camera.json → {out / 'camera.json'}")

    # ── depth + normals ───────────────────────────────────────────────────────
    normals_all: list[np.ndarray] = []
    for i in range(S):
        info   = prep_info[i] if i < len(prep_info) else {}
        H_o    = info.get("H_o", H_proc)
        W_o    = info.get("W_o", W_proc)
        new_h  = info.get("new_h", H_proc)
        crop_y = info.get("crop_y", 0)

        depth_p = depth_all[i, :, :, 0]    # (H_proc, W_proc) in proc space

        # Normals in proc space (intrinsics are accurate there), then rotate
        normals_cam  = _normals_from_depth(depth_p, intrinsics[i])
        R_c2w        = extrinsics[i, :3, :3].T          # cam→world rotation
        normals_w_p  = normals_cam @ R_c2w.T            # (H_proc, W_proc, 3)

        up_kw = dict(H_proc=depth_p.shape[0], new_h=new_h, new_w=W_proc,
                     crop_y=crop_y, H_o=H_o, W_o=W_o)

        depth_o   = _upsample_to_orig(depth_p,    **up_kw, is_depth=True)
        normals_o = _upsample_to_orig(normals_w_p, **up_kw, is_depth=False)
        n_norm    = np.linalg.norm(normals_o, axis=-1, keepdims=True)
        normals_o /= np.maximum(n_norm, 1e-8)

        normals_all.append(normals_o)

        np.save(str(out / f"depth_{i}.npy"), depth_o)
        Image.fromarray(_colorize_depth(depth_o)).save(str(out / f"depth_{i}.png"))

        np.save(str(out / f"normal_{i}.npy"), normals_o)
        Image.fromarray(_normals_to_rgb(normals_o)).save(str(out / f"normal_{i}.png"))

        print(f"[vggt]   frame {i}: {W_o}×{H_o}px  (proc {W_proc}×{H_proc})")

    # ── point cloud ───────────────────────────────────────────────────────────
    pts_flat  = world_pts.reshape(-1, 3)
    conf_flat = world_conf.reshape(-1)
    cols_flat = (images_np.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    ply_path  = str(out / "points.ply")
    _write_ply(ply_path, pts_flat, cols_flat, conf=conf_flat, conf_pct=conf_pct)
    kept = int((conf_flat >= np.percentile(conf_flat, conf_pct)).sum())
    print(f"[vggt] points.ply → {ply_path}  ({kept:,} pts)")

    print(f"[vggt] Done → {out}/")
    return dict(camera=cam_data, depth=depth_all, normals=normals_all,
                world_points=world_pts, extrinsics=extrinsics, intrinsics=intrinsics)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run VGGT (HARMONY wrapper)")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out",      default="outputs/vggt")
    ap.add_argument("--conf-pct", type=float, default=10.0,
                    help="Filter lowest N%% confidence points (default 10)")
    ap.add_argument("--device",   default=None)
    args = ap.parse_args()
    run(args.images, out_dir=args.out, conf_pct=args.conf_pct, device=args.device)
