"""bake_carpet.py — bake the (2D-only) carpet into a 3D textured floor plane.

The furniture phase places carpets as a flat 2D overlay (is_carpet, glb_path=None),
so the path-traced Blender scene has no rug. This builds a textured quad on the
floor from the carpet's segmentation crop+mask and the camera, writing
<scene>/lightings/objects/carpet.glb, which render_blender picks up automatically.

    python lighting_module/bake_carpet.py --scene outputs/.../living_room8
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from pygltflib import GLTF2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from object_placement.furniture.place_furniture_vggt import _camera_axes


def bake(scene_dir: str) -> str | None:
    D = scene_dir
    pl = json.load(open(f"{D}/furniture/furniture_placements.json"))
    carp = [e for e in pl if e.get("is_carpet")]
    if not carp:
        print("[carpet] no is_carpet placement — nothing to bake")
        return None
    # [x0, y0, x1, y1] in the reference image. Older placements store it as
    # box_px; the FRONT3D/_HARMONY300 carpet entries use _mask_bbox instead.
    box = carp[0].get("box_px") or carp[0].get("_mask_bbox")
    if box is None:
        print("[carpet] no box_px/_mask_bbox on carpet placement — cannot bake")
        return None

    crop_p = sorted(glob.glob(f"{D}/furniture/segmented/segment_*_carpet_crop.png"))
    mask_p = sorted(glob.glob(f"{D}/furniture/segmented/segment_*_carpet_mask.png"))
    if not crop_p or not mask_p:
        print("[carpet] no carpet crop/mask segmentation found")
        return None
    crop = Image.open(crop_p[-1]).convert("RGB")
    mask_full = Image.open(mask_p[-1]).convert("L")
    # The furniture segmenter pads carpet crops onto a square canvas (gray
    # fill, sized for the 3D-gen path) even though carpets never go through
    # 3D-gen. If the box_px aspect ratio doesn't match the crop's, the crop is
    # padded — tighten to the real content band, or resizing the box-cropped
    # mask onto the full padded canvas below stretches/misaligns the alpha.
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    if box_w > 0 and box_h > 0 and abs((box_w / box_h) / (crop.width / crop.height) - 1) > 0.3:
        carr = np.asarray(crop)
        gray = ((np.abs(carr[..., 0].astype(int) - carr[..., 1].astype(int)) < 3) &
                (np.abs(carr[..., 1].astype(int) - carr[..., 2].astype(int)) < 3))
        rows = np.where(gray.mean(axis=1) < 0.9)[0]
        cols = np.where(gray.mean(axis=0) < 0.9)[0]
        if len(rows) > 5 and len(cols) > 5:
            crop = crop.crop((int(cols.min()), int(rows.min()),
                               int(cols.max()) + 1, int(rows.max()) + 1))
            print(f"[carpet] padded-canvas crop detected → tightened to real content {crop.size}")
    # The crop is the box region; the mask is often the FULL reference image.
    # Crop the mask to the carpet box so its alpha aligns with the crop.
    if mask_full.size != crop.size:
        bx = [int(v) for v in box]
        mask_full = mask_full.crop((bx[0], bx[1], bx[2], bx[3]))
    mask = mask_full.resize(crop.size)

    # ── Robust rug-pixel detection (handles rug-on-white "masks") ─────────────
    # The carpet 'mask' PNG is USUALLY a binary alpha (rug=white on black). But
    # for some scenes it is the rug isolated on a WHITE/grey background (no black
    # bg at all). The stock ``mask > 10`` test then treats that white background as
    # opaque rug → the whole quad renders white (with a stray textured stripe).
    # Detect the rug-on-white case (≈no true-black pixels) and: (a) derive the rug
    # mask from the CROP's colour instead of the grey level, and (b) auto-enable
    # TILE mode so a clean, coherent rug patch (real colours from the crop) is
    # tiled across the floor rectangle rather than baking the unreliable outline.
    _mask_L = np.asarray(mask.convert("L"))
    _rug_on_white = (_mask_L < 20).mean() < 0.05 and (_mask_L > 235).mean() > 0.20

    def _rug_bool(mask_img, crop_img):
        marr = np.asarray(mask_img.convert("L"))
        if (marr < 20).mean() > 0.12:            # real black background → binary mask
            return marr > 10
        hsv = np.asarray(crop_img.convert("HSV")).astype(np.int32)
        return (hsv[..., 1] > 35) | (hsv[..., 2] < 215)   # coloured OR non-bright

    if _rug_on_white and not os.environ.get("SCENEWEAVE_CARPET_TILE"):
        os.environ["SCENEWEAVE_CARPET_TILE"] = "1"
        print("[carpet] rug-on-white mask detected → TILE mode (rug from crop colour)")

    # RECT mode (DEFAULT): the rug must render as a clean 4-edge RECTANGLE that
    # extends to cover the whole back-projected region — it should NOT be warped to
    # follow the back-projected mask's specific (perspective-twisted) outline. So we
    # make the whole quad opaque and let the axis-aligned rect geometry (computed
    # below from the back-projected mask extent) define the rug footprint. Opt back
    # into the old mask-shaped alpha with SCENEWEAVE_CARPET_MASKSHAPE=1.
    rect_mode = os.environ.get("SCENEWEAVE_CARPET_MASKSHAPE") != "1"
    # TILE mode: the rug is heavily occluded (glass table / furniture) so its
    # mask is fragmented into chunks — baking it directly gives a "shattered"
    # rug with floor showing through. The rug is a continuous, repeating pattern
    # in reality, so extract the cleanest densely-covered rug patch and TILE it
    # across the whole opaque rectangle → one coherent rug on the floor.
    if os.environ.get("SCENEWEAVE_CARPET_TILE"):
        m0 = _rug_bool(mask, crop)
        H0, W0 = m0.shape
        pw, ph = max(8, W0 // 4), max(8, H0 // 4)
        # integral image → densest pw×ph window of rug coverage
        ii = np.zeros((H0 + 1, W0 + 1), np.int64)
        ii[1:, 1:] = np.cumsum(np.cumsum(m0.astype(np.int64), 0), 1)
        best, bxy = -1, (0, 0)
        for yy in range(0, H0 - ph, max(1, ph // 4)):
            for xx in range(0, W0 - pw, max(1, pw // 4)):
                s = (ii[yy + ph, xx + pw] - ii[yy, xx + pw]
                     - ii[yy + ph, xx] + ii[yy, xx])
                if s > best:
                    best, bxy = s, (xx, yy)
        px, py = bxy
        patch = crop.crop((px, py, px + pw, py + ph))
        # tile the patch across the full crop-sized canvas
        tiled = Image.new("RGB", crop.size)
        for ty in range(0, crop.size[1], ph):
            for tx in range(0, crop.size[0], pw):
                tiled.paste(patch, (tx, ty))
        crop = tiled
        mask = Image.new("L", crop.size, 255)  # opaque rectangle
        rect_mode = True  # skip the fragmented-mask fill paths below
        print(f"[carpet] TILE: patch {pw}x{ph} @ {bxy} tiled over {crop.size}")
    # MASKSHAPE (default preferred): keep the rug's TRUE footprint — only rug
    # pixels show (correct colour, matches the 2D furniture overlay), just close
    # small furniture-occlusion gaps.  No convex hull (→ polygon) and no bbox
    # rectangle (→ picks up surrounding floor).  Disable with SCENEWEAVE_CARPET_HULL.
    maskshape = not rect_mode and not os.environ.get("SCENEWEAVE_CARPET_HULL")
    m = _rug_bool(mask, crop)
    ys, xs = np.where(m)
    if rect_mode:
        # Opaque rectangle, but the crop's non-rug pixels (wood floor bleeding in
        # at a corner where the rug's box isn't fully covered) would render as a
        # brown patch on the rug. Fill non-rug pixels with the rug's MEDIAN colour
        # so only the rug (its pattern where visible) shows across the rectangle.
        if m.sum() > 10:
            arr = np.asarray(crop).copy()
            med = np.median(arr[m], axis=0).astype(arr.dtype)
            arr[~m] = med
            crop = Image.fromarray(arr)
        mask = Image.new("L", crop.size, 255)
    elif maskshape and len(xs) > 10:
        try:
            from scipy import ndimage
            filled = ndimage.binary_closing(m, np.ones((15, 15)))
            filled = ndimage.binary_fill_holes(filled)
            mask = Image.fromarray((filled * 255).astype(np.uint8))
        except Exception as e:
            print("[carpet] maskshape fill skipped:", e)
    # The rug is occluded by furniture in the photo, so its segmentation mask is
    # fragmented — rendering it directly shows wood floor through the gaps
    # ("shattered" rug). The rug is continuous UNDER the furniture in 3D, so fill
    # the mask into one coherent shape (convex hull of the rug pixels).
    elif len(xs) > 10:
        pts = np.column_stack([xs, ys]).astype(np.int32)
        filled = None
        try:
            import cv2
            hull = cv2.convexHull(pts)
            filled = np.zeros(m.shape, np.uint8)
            cv2.fillConvexPoly(filled, hull, 255)
        except Exception:
            try:
                from scipy import ndimage
                k = np.ones((25, 25))
                filled = (ndimage.binary_fill_holes(
                    ndimage.binary_closing(m, k)) * 255).astype(np.uint8)
            except Exception as e:
                print("[carpet] mask-fill skipped:", e)
        if filled is not None:
            mask = Image.fromarray(filled)
    rgba = crop.convert("RGBA")
    rgba.putalpha(mask)

    cam = json.load(open(f"{D}/camera_vggt.json"))
    W, H = int(cam["width_px"]), int(cam["height_px"])
    cp = np.array(cam["position_m"], float)
    la = np.array(cam["look_at_m"], float)
    up = np.array(cam.get("up", [0, 1, 0]), float)
    r, uc, fw = _camera_axes(cp, la, up)
    fx = W / (2 * np.tan(np.radians(cam["hfov_deg"] / 2)))
    cx, cy = W / 2.0, H / 2.0

    def backproject_to_floor(px, py):
        dc = np.array([(px - cx) / fx, (cy - py) / fx, 1.0])
        dw = dc[0] * r + dc[1] * uc + dc[2] * fw
        dw = dw / np.linalg.norm(dw)
        t = -cp[1] / dw[1]            # intersect floor y=0
        return cp + t * dw

    # Build an AXIS-ALIGNED rectangle on the floor: sides parallel to the room
    # walls (which lie along world X and Z). Back-projecting the four image-bbox
    # corners instead gives a perspective TRAPEZOID (the rug looks skewed / non-
    # rectangular). So back-project the rug MASK pixels to the floor and take
    # their world-XZ extent as the rug's rectangle.
    rect = None
    # The furniture-placement stage already computes an axis-aligned floor
    # footprint for the carpet (from the full 2D overlay, not just the SAM
    # box) — prefer it. The mask-back-projection fallback below only sees
    # rug pixels inside box_px, which for a heavily-occluded rug undershoots
    # the true footprint (and its "mask is full-image-sized" branch treats
    # the mask's white background as rug, corrupting the fit further).
    wc = carp[0].get("world_corners")
    if wc:
        xs_wc = [p[0] for p in wc]
        zs_wc = [p[2] for p in wc]
        rect = (min(xs_wc), max(xs_wc), min(zs_wc), max(zs_wc))
        print(f"[carpet] using placement world_corners footprint x[{rect[0]:.2f},{rect[1]:.2f}] "
              f"z[{rect[2]:.2f},{rect[3]:.2f}]")
    if rect is None:
        try:
            mm = np.asarray(Image.open(mask_p[-1]).convert("L"))
            if mm.shape[:2] != (H, W):
                # mask is crop-sized → map its rug pixels back to image coords via box
                ys_c, xs_c = np.where(mm > 10)
                bx0, by0, bx1, by1 = [float(v) for v in box]
                ix = bx0 + xs_c / max(1, mm.shape[1]) * (bx1 - bx0)
                iy = by0 + ys_c / max(1, mm.shape[0]) * (by1 - by0)
            else:
                ys_c, xs_c = np.where(mm > 10)
                ix, iy = xs_c.astype(float), ys_c.astype(float)
            if len(ix) > 5000:
                sel = np.linspace(0, len(ix) - 1, 5000).astype(int)
                ix, iy = ix[sel], iy[sel]
            pts = []
            for px, py in zip(ix, iy):
                dc = np.array([(px - cx) / fx, (cy - py) / fx, 1.0])
                dw = dc[0] * r + dc[1] * uc + dc[2] * fw
                dw = dw / np.linalg.norm(dw)
                if dw[1] >= -1e-3:           # ray parallel to / above floor → skip
                    continue
                t = -cp[1] / dw[1]
                if t <= 0 or t > 60:         # reject horizon blow-ups
                    continue
                p = cp + t * dw
                pts.append([p[0], p[2]])
            pts = np.array(pts, float)
            if len(pts) >= 20:
                # robust extent (percentiles reject stray back-projected outliers)
                xmin, xmax = np.percentile(pts[:, 0], [1, 99])
                zmin, zmax = np.percentile(pts[:, 1], [1, 99])
                rect = (float(xmin), float(xmax), float(zmin), float(zmax))
        except Exception as e:
            print("[carpet] axis-aligned rect-fit skipped:", e)

    if rect is not None:
        xmin, xmax, zmin, zmax = rect
        verts = np.array([[xmin, 0.012, zmin], [xmax, 0.012, zmin],
                          [xmax, 0.012, zmax], [xmin, 0.012, zmax]], float)
        print(f"[carpet] axis-aligned rect x[{xmin:.2f},{xmax:.2f}] "
              f"z[{zmin:.2f},{zmax:.2f}]")
    else:
        x0, y0, x1, y1 = box
        verts = np.array([backproject_to_floor(x0, y0), backproject_to_floor(x1, y0),
                          backproject_to_floor(x1, y1), backproject_to_floor(x0, y1)],
                         float)
        verts[:, 1] = 0.012           # lift just off the floor to avoid z-fighting
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=rgba)

    out_dir = f"{D}/lightings/objects"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/carpet.glb"
    mesh.export(out)

    # Patch material: alpha from the mask (only the rug shows, not the crop's
    # surrounding floor), double-sided so it's lit from above regardless of winding.
    g = GLTF2().load(out)
    for m in g.materials:
        m.alphaMode = "BLEND"
        m.doubleSided = True
        if m.pbrMetallicRoughness:
            # trimesh's TextureVisuals export defaults baseColorFactor to its grey
            # material colour (~0.4) which MULTIPLIES the texture → the rug renders
            # ~40% dark ("texture missing"). Reset to white so the texture shows.
            m.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
            m.pbrMetallicRoughness.roughnessFactor = 0.95
            m.pbrMetallicRoughness.metallicFactor = 0.0
    g.save(out)
    print(f"[carpet] baked → {out}  corners(x,z)="
          f"{[(round(v[0],2),round(v[2],2)) for v in verts]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    bake(a.scene)
