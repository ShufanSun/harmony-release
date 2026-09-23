"""
floorplan/texture_cleanup.py — deterministic defect removal + delighting for
backprojected wall textures.

A backprojected wall texture should be "mostly one material". Regions that
deviate strongly from that dominant material — a cabinet baked in, floor/
baseboard bleed at the bottom, colour bleed from a neighbouring wall — are
defects. We:

  1. detect_defects : flag pixels far from the dominant material (robust median
     in Lab), clean the mask morphologically.
  2. delight        : flatten the baked-in lighting gradient so tone is uniform
     (flat albedo) — fixes the "dimmer / uneven" look.
  3. clean          : delight, then complete the defect regions with mirrored
     wall pattern so panel stripes carry through instead of a flat smear.

No generative model — pure OpenCV, deterministic.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_defects(bgr: np.ndarray, delta_thresh: float = 20.0,
                   dark_drop: float = 30.0) -> np.ndarray:
    """uint8 mask (255 = defect) of pixels deviating from the dominant material.
    Dominant material = per-channel median in Lab (robust while the wall
    material occupies >50% of the map, which holds for these bakes)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    med = np.median(lab.reshape(-1, 3), axis=0)

    dist = np.sqrt(((lab - med) ** 2).sum(axis=2))
    defect = dist > delta_thresh
    defect |= lab[..., 0] < (med[0] - dark_drop)        # markedly darker → cabinet/baseboard

    mask = defect.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9, iterations=2)
    mask = cv2.dilate(mask, k3, iterations=1)
    return mask


def protect_edge_trim(mask: np.ndarray, edge_frac: float = 0.10,
                      row_cover: float = 0.35) -> np.ndarray:
    """Un-flag legitimate architectural trim — a baseboard/skirting at the
    floor-wall edge or a crown moulding / dark line at the wall-ceiling edge.

    In a thin band at the very top and bottom of the texture (the trim zones),
    any ROW whose flagged-fraction exceeds ``row_cover`` is treated as a
    horizontal trim line and preserved (cleared from the mask). A cabinet only
    occupies part of a row's width, so it stays flagged; a full-width dark line
    does not. Interior defects (the cabinet body) are untouched."""
    H, W = mask.shape
    band = max(1, int(edge_frac * H))
    out = mask.copy()
    cover = (mask > 0).mean(axis=1)                 # per-row flagged fraction
    for y in list(range(band)) + list(range(H - band, H)):
        if cover[y] >= row_cover:
            out[y, :] = 0                           # keep this trim row
    return out


def shear_angle(bgr: np.ndarray, min_deg: float = 2.0, max_deg: float = 15.0,
                min_lines: int = 3, max_std: float = 2.5) -> float:
    """Detect a vertical-shear tilt in a wall texture: rectifying an oblique wall
    whose top corner is off-frame leaves verticals vertical but tilts horizontal
    features (panel/trim lines) by a uniform angle. Returns that angle in degrees
    (signed), or 0.0 if there isn't a clear, consistent tilt (so frontal/plain
    walls are untouched)."""
    H, W = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=35,
                            minLineLength=int(W * 0.20), maxLineGap=15)
    angs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < abs(y2 - y1):
                continue                                   # keep horizontal-ish
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angs.append((a + 90) % 180 - 90)
    if len(angs) < min_lines:
        return 0.0
    angs = np.asarray(angs)
    theta = float(np.median(angs))
    if float(np.std(angs)) > max_std or not (min_deg <= abs(theta) <= max_deg):
        return 0.0
    return theta


def apply_vshear(arr: np.ndarray, theta_deg: float, border_value=0,
                 nearest: bool = False) -> np.ndarray:
    """Undo a vertical shear of theta_deg (verticals preserved, horizontals
    leveled). Used on the texture and, with border_value=1/nearest, on the
    invalid/stretch masks so newly-exposed border becomes inpaint-able."""
    if abs(theta_deg) < 1e-6:
        return arr
    H, W = arr.shape[:2]
    s = np.tan(np.radians(theta_deg))
    M = np.float32([[1, 0, 0], [-s, 1, s * W / 2]])
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4
    return cv2.warpAffine(arr, M, (W, H), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def straighten_trim(bgr: np.ndarray, edge_frac: float = 0.12,
                    dark_drop: float = 25.0) -> np.ndarray:
    """Replace a warped/wedge-shaped edge trim with a clean, THIN uniform
    horizontal stripe. A slanted junction projects to a wedge (thick on one
    side); filling its whole extent makes an over-wide bar. Instead we clear the
    wedge to wall colour and draw a stripe whose thickness is the trim's
    consistent (low-percentile) thickness, anchored at the band's start."""
    H, W, _ = bgr.shape
    band = max(4, int(edge_frac * H))
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    wall = float(np.median(L))
    dark = L[:band] < (wall - dark_drop)
    cover = dark.mean(axis=1)                      # per-row dark fraction
    rows = np.where(cover > 0.25)[0]
    if rows.size < 2:
        return bgr                                 # no clear trim line
    r0 = int(rows.min())

    counts = dark.sum(axis=0)                       # per-column dark thickness
    has = counts[counts > 2]
    if has.size < 3:
        return bgr
    # consistent (thin) thickness — low percentile so the wedge's thick end
    # doesn't set the width; clamp to a sane baseboard range.
    t = int(np.clip(np.percentile(has, 25), 4, max(6, band // 3)))

    trim_col = np.median(bgr[:band][dark].reshape(-1, 3), axis=0).astype(np.uint8)
    wall_col = np.median(bgr.reshape(-1, 3), axis=0).astype(np.uint8)

    out = bgr.copy()
    wedge = np.zeros((H, W), dtype=bool)
    wedge[:band] = dark
    out[wedge] = wall_col                           # erase the whole wedge → wall
    out[r0:r0 + t] = trim_col                        # thin uniform stripe
    return out


def detect_openings(bgr: np.ndarray, valid: "np.ndarray | None" = None,
                    drop: float = 80.0, min_area_frac: float = 0.015,
                    which: str = "dark", protect_trim: bool = True) -> np.ndarray:
    """Locate large NON-surface regions whose LIGHTNESS deviates strongly from
    the surface material, so they can be re-filled with the surface pattern:
      • which='dark'   — far darker than the surface: a doorway/opening to dark
                         space, or dark wall-bleed onto a light floor.
      • which='bright' — far brighter: a ceiling lamp/fixture sampled onto a
                         dark wall.
      • which='both'   — either.
    Lightness-keyed so a coloured accent strip/panel (different hue, SAME
    lightness) is NOT flagged. Thin grooves are removed by an opening morphology;
    only large connected blobs survive.

    protect_trim keeps wide dark bands at the edges (baseboard/crown) — use for
    walls; disable for the floor, where edge darkness is wall-bleed, not trim.
    Returns a uint8 mask (1 = region to re-fill)."""
    H, W = bgr.shape[:2]
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    ref = L[valid > 0] if valid is not None and np.any(valid) else L.reshape(-1)
    med = float(np.median(ref))

    if which == "bright":
        anom = (L > med + drop)
    elif which == "both":
        anom = (L < med - drop) | (L > med + drop)
    else:
        anom = (L < med - drop)
    anom = anom.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    anom = cv2.morphologyEx(anom, cv2.MORPH_OPEN, k, iterations=2)   # drop grooves

    n, labels, stats, _ = cv2.connectedComponentsWithStats(anom, 8)
    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * H * W:
            out[labels == i] = 1
    if out.any():
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if protect_trim:
            out = (protect_edge_trim(out * 255) > 0).astype(np.uint8)
    return out


def delight(bgr: np.ndarray, mask: np.ndarray, sigma: float = 41.0) -> np.ndarray:
    """Flatten the low-frequency lighting gradient → uniform tone. Illumination
    is estimated from a defect-filled base so the dark cabinet doesn't bias it;
    chroma and high-frequency texture are preserved."""
    base = cv2.inpaint(bgr, mask, 11, cv2.INPAINT_NS)
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    illum = cv2.GaussianBlur(L, (0, 0), sigma)
    target = float(np.median(illum))
    lab[..., 0] = np.clip(L * target / np.maximum(illum, 1.0), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _fill_stretch(out: np.ndarray, stretch_mask) -> np.ndarray:
    """Reconstruct the over-stretched grazing-corner region (located
    geometrically by the caller) with mirrored wall pattern + a short TELEA
    blend. Local to that corner; leaves the rest of the texture untouched."""
    if stretch_mask is None or not np.any(stretch_mask):
        return out
    sm = (stretch_mask > 0).astype(np.uint8)
    sm = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    seeded = out.copy()
    mir = np.fliplr(out)
    m = sm > 0
    seeded[m] = mir[m]
    return cv2.inpaint(seeded, sm, 3, cv2.INPAINT_TELEA)


def clean(bgr: np.ndarray, delta_thresh: float = 20.0,
          dark_drop: float = 30.0, keep_edge_trim: bool = True,
          stretch_mask: "np.ndarray | None" = None,
          light: bool = True, return_mask: bool = False):
    """Finish a backprojected wall texture.

    light=True (default): PRESERVE the backprojected pattern — do NOT delight,
    do NOT remove "defects" (which on a faithful empty room are real features
    like accent strips/shading), do NOT redraw trim. Only fill the geometrically
    located over-stretched grazing corner. This is the "just complete the
    backprojected texture" path.

    light=False: the aggressive pass (delight + colour-defect removal + edge-trim
    straightening). Use only when the source still has occluders/lighting issues;
    it can rewrite real texture, so it is no longer the default. Also used as the
    contrast-enhanced *probe* for shear detection.

    keep_edge_trim (light=False only) preserves baseboard/crown lines.
    stretch_mask: over-stretched grazing-corner pixels to reconstruct.
    """
    if light:
        # Preserve ALL backprojected pixels as-is; only the completion step (run
        # by the caller) fills genuinely-missing regions. We deliberately do NOT
        # run the stretch-fix here — on oblique walls it flags real (low-res but
        # valid) grazing pixels as "over-stretched" and mirror-blurs them, which
        # is exactly the glaze near the wall line. Keep the real data sharp.
        out = bgr.copy()
        if return_mask:
            return out, np.zeros(bgr.shape[:2], dtype=np.uint8)
        return out

    mask = detect_defects(bgr, delta_thresh, dark_drop)
    if keep_edge_trim:
        mask = protect_edge_trim(mask)
    delit = delight(bgr, mask)

    # Smooth flat fill: seed large holes with the dominant material colour so
    # Navier-Stokes has a plausible base, then inpaint. This keeps the filled
    # regions clean and uniform (consistent with low-data walls) instead of the
    # wavy artifacts a mirror-seed produces.
    lab = cv2.cvtColor(delit, cv2.COLOR_BGR2LAB).astype(np.float32)
    med_lab = np.median(lab.reshape(-1, 3), axis=0).astype(np.uint8)
    med_bgr = cv2.cvtColor(med_lab.reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]

    seeded = delit.copy()
    if float((mask > 0).mean()) > 0.05:
        seeded[mask > 0] = med_bgr
        radius = 25
    else:
        radius = 9
    out = cv2.inpaint(seeded, mask, radius, cv2.INPAINT_NS)

    out = _fill_stretch(out, stretch_mask)

    if keep_edge_trim:
        out = straighten_trim(out)

    return (out, mask) if return_mask else out
