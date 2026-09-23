"""
rectify_windows.py — perspective-correct inpainted window images to canonical
                     front-facing rectangular view.

For each window segment in segment_results.json:
  1. Load the inpainted PNG (from inpainted/).
  2. Use SAM (box-prompted with the foreground bbox) to get a precise window mask.
  3. Find the convex hull of the mask → 4-corner approximation via minAreaRect.
  4. Apply homography to warp the perspective quad to a clean axis-aligned rectangle.
  5. Re-pad with grey background and overwrite the inpainted PNG (or save as
     a new file if --suffix is given).

Falls back to simple foreground-bbox crop if SAM is unavailable or fails.

Usage:
    python -m object_placement.wall_mounted.rectify_windows \\
        --output-dir outputs/living_room9

    # Save as separate files instead of overwriting:
    python -m object_placement.wall_mounted.rectify_windows \\
        --output-dir outputs/living_room9 --suffix _rect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GSA_ROOT  = _REPO_ROOT / "Grounded-Segment-Anything"
sys.path.insert(0, str(_GSA_ROOT / "segment_anything"))

_SAM_CKPT      = _GSA_ROOT / "weights/sam_vit_h_4b8939.pth"
_SAM_MODEL_TYPE = "vit_h"

_BG_GREY = (185, 185, 185)
_BG_TOL  = 25


# ── SAM loader (cached) ───────────────────────────────────────────────────────

_sam_predictor = None

def _get_sam(device: str = "cuda"):
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import SamPredictor, sam_model_registry
        print("[rectify] Loading SAM …")
        sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=str(_SAM_CKPT))
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _foreground_bbox(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (r0, r1, c0, c1) tight bounding box of non-grey pixels, or None."""
    grey  = np.array(_BG_GREY, dtype=np.float32)
    is_fg = (np.abs(arr.astype(np.float32) - grey) > _BG_TOL).any(axis=2)
    rows  = np.where(is_fg.any(axis=1))[0]
    cols  = np.where(is_fg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _sam_mask(arr: np.ndarray, bbox: tuple[int,int,int,int],
              device: str = "cuda") -> np.ndarray | None:
    """Run SAM with a box prompt around the foreground bbox.
    Returns the best binary mask (H×W bool), or None on failure.
    """
    try:
        predictor = _get_sam(device)
        predictor.set_image(arr)
        r0, r1, c0, c1 = bbox
        box = np.array([c0, r0, c1, r1], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)
    except Exception as e:
        print(f"  [rectify] SAM failed: {e}")
        return None


def _quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Approximate the window outline as a 4-point quad from the SAM mask.

    Uses the convex hull of the mask contour, then approximates to 4 corners
    via minAreaRect. Returns (4,2) float32 in [TL, TR, BR, BL] order.
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour
    contour = max(contours, key=cv2.contourArea)
    hull    = cv2.convexHull(contour)

    # Approximate to polygon — try to get exactly 4 corners
    peri = cv2.arcLength(hull, True)
    for eps_frac in [0.02, 0.04, 0.06, 0.08, 0.10]:
        approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(approx) == 4:
            break
    else:
        # Fall back to minAreaRect
        rect   = cv2.minAreaRect(contour)
        approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.int32)

    pts = approx.reshape(-1, 2).astype(np.float32)

    # Order: TL, TR, BR, BL
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify(img: Image.Image, device: str = "cuda") -> Image.Image:
    """Segment with SAM, find 4-corner quad, warp to canonical rectangle."""
    arr  = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]

    # ── Step 1: foreground bbox ───────────────────────────────────────────────
    bbox = _foreground_bbox(arr)
    if bbox is None:
        print("  [rectify] no foreground found — returning unchanged")
        return img
    r0, r1, c0, c1 = bbox

    # ── Step 2: SAM mask ──────────────────────────────────────────────────────
    mask = _sam_mask(arr, bbox, device=device)

    if mask is not None:
        quad = _quad_from_mask(mask)
        print(f"  [rectify] SAM mask obtained, quad={np.round(quad,1).tolist() if quad is not None else None}")
    else:
        quad = None

    # ── Step 3: warp to rectangle ─────────────────────────────────────────────
    if quad is not None:
        # Target size: use the axis-aligned bbox dimensions
        dst_w = c1 - c0
        dst_h = r1 - r0
        dst   = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)

        M      = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(arr, M, (dst_w, dst_h),
                                     flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=_BG_GREY)
    else:
        # Fallback: tight bbox crop (already axis-aligned, no perspective fix)
        print("  [rectify] quad unavailable — using bbox crop")
        warped = arr[r0:r1, c0:c1]

    # ── Step 4: re-pad with grey to original canvas size ─────────────────────
    result = np.full((H, W, 3), _BG_GREY, dtype=np.uint8)
    ph = max(0, (H - warped.shape[0]) // 2)
    pw = max(0, (W - warped.shape[1]) // 2)
    result[ph:ph+warped.shape[0], pw:pw+warped.shape[1]] = warped

    return Image.fromarray(result)


def run(output_dir: str | Path, suffix: str = "", device: str = "cuda") -> None:
    out_dir     = Path(output_dir) / "wall_mounted"
    inpaint_dir = out_dir / "inpainted"
    results_p   = out_dir / "segment_results.json"

    with open(results_p) as f:
        data = json.load(f)

    for seg in data.get("segments", []):
        if seg.get("type") not in ("window", "door", "art"):
            continue
        inpaint_file = seg.get("inpaint_file")
        if not inpaint_file:
            continue
        src = inpaint_dir / inpaint_file
        if not src.exists():
            print(f"[rectify] {src.name} not found — skipping")
            continue

        print(f"\n[rectify] {src.name}")
        img       = Image.open(src).convert("RGB")
        rectified = rectify(img, device=device)

        if suffix:
            stem = Path(inpaint_file).stem
            dst  = inpaint_dir / f"{stem}{suffix}.png"
            # Update segment record to point to new file
            seg["inpaint_file"] = dst.name
        else:
            dst = src   # overwrite

        rectified.save(dst)
        print(f"  saved → {dst.name}")

    if suffix:
        with open(results_p, "w") as f:
            json.dump(data, f, indent=2)
        print("\n[rectify] segment_results.json updated with new filenames")

    print("\n[rectify] Done.")


def main():
    ap = argparse.ArgumentParser(
        description="Rectify inpainted window images to canonical rectangular view."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output directory (contains wall_mounted/)")
    ap.add_argument("--suffix", default="",
                    help="If set, save as <name><suffix>.png instead of overwriting. "
                         "E.g. --suffix _rect")
    ap.add_argument("--device", default="cuda",
                    help="Device for SAM (cuda or cpu)")
    args = ap.parse_args()
    run(output_dir=args.output_dir, suffix=args.suffix, device=args.device)


if __name__ == "__main__":
    main()
