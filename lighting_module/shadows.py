"""Shadow pass for the lit preview.

The base renderer is purely additive: per-light Gaussian glows brighten
the regions near each emitter. To make objects feel grounded and to give
the lighting some directionality, this module adds two kinds of darken
passes on top:

  1. CONTACT SHADOW
     A small soft dark blob at every placed object's floor base point.
     Always present (it's the floor occlusion of the object's own footprint).

  2. CAST SHADOW
     For each (caster × active key-light) pair, an elongated dark ellipse
     stretched away from the light along the floor. Length scales with
     object height and the light's elevation; capped so we never get
     runaway sausage shadows.

Casters come from the placement JSONs of furniture, decorations, and
floor-adjacent wall-mounted objects. Light sources come from the
`lights` dict (active point lights) plus daylight-on windows treated as
directional sources whose direction is the wall normal pointing into the
room.

Output: nothing — `add_shadows` mutates the buffer in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# How dark a fully-saturated shadow gets (1.0 = pure black, 0.0 = no effect).
_CONTACT_DARKEN = 0.30
_CAST_DARKEN_PER_INTENSITY = 0.55
_MAX_CAST_LENGTH_FACTOR = 3.0          # shadow length ≤ 3× object footprint
_MIN_CAST_LENGTH_M = 0.20              # ignore shadows shorter than this


def _project(world_pt: np.ndarray, cam_pos, right_v, up_c_v, fwd_v,
             fx: float, cx: float, cy: float):
    d = world_pt - cam_pos
    xc = float(np.dot(d, right_v))
    yc = float(np.dot(d, up_c_v))
    zc = float(np.dot(d, fwd_v))
    if zc <= 0.01:
        return None
    return cx + fx * xc / zc, cy - fx * yc / zc, zc


def _read_placements(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("placements") or data.get("placement_order") or []
    return data if isinstance(data, list) else []


def collect_casters(output_dir: str | Path) -> list[dict]:
    """Return a list of {position_m, size_m, name} for every placed object that
    rests on the floor (or on a surface) and is therefore worth shadowing."""
    out_dir = Path(output_dir)
    casters: list[dict] = []

    for jpath in (out_dir / "furniture" / "furniture_placements.json",
                  out_dir / "decorations" / "placements" / "decoration_placements.json"):
        for e in _read_placements(jpath):
            pos = e.get("position_m")
            size = e.get("size_m") or {}
            if not pos or not size:
                continue
            name = e.get("phrase") or e.get("type") or "obj"
            casters.append({
                "name": str(name),
                "position_m": [float(v) for v in pos],
                "size_m": {
                    "width_m":  float(size.get("width_m",  0.30)),
                    "height_m": float(size.get("height_m", 0.30)),
                    "depth_m":  float(size.get("depth_m",  0.30)),
                },
            })
    return casters


def _draw_dark_ellipse(buf: np.ndarray, screen_pts: list[tuple[float, float]],
                       darken_factor: float) -> None:
    """Fit an axis-aligned ellipse to `screen_pts` and darken pixels inside
    it with a soft Gaussian falloff.

    `darken_factor` ∈ [0, 1]: 0 = no effect, 1 = pure black at the centre.
    """
    if darken_factor <= 1e-3 or not screen_pts:
        return
    xs = [p[0] for p in screen_pts]
    ys = [p[1] for p in screen_pts]
    cx_s = (min(xs) + max(xs)) * 0.5
    cy_s = (min(ys) + max(ys)) * 0.5
    rx = max((max(xs) - min(xs)) * 0.5, 4.0)
    ry = max((max(ys) - min(ys)) * 0.5, 3.0)

    H, W, _ = buf.shape
    x0 = int(max(0, cx_s - rx * 1.6)); x1 = int(min(W, cx_s + rx * 1.6))
    y0 = int(max(0, cy_s - ry * 1.6)); y1 = int(min(H, cy_s + ry * 1.6))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx = (xx - cx_s) / max(rx, 1.0)
    ny = (yy - cy_s) / max(ry, 1.0)
    norm = nx * nx + ny * ny
    mask = np.exp(-1.6 * norm).astype(np.float32)
    mask = np.clip(mask, 0.0, 1.0) * float(darken_factor)

    sub = buf[y0:y1, x0:x1].astype(np.float32)
    sub *= (1.0 - mask[..., None])
    np.clip(sub, 0.0, 255.0, out=sub)
    buf[y0:y1, x0:x1] = sub.astype(np.uint8)


def _add_contact_shadow(buf: np.ndarray, caster: dict, cam) -> None:
    p = np.array(caster["position_m"], dtype=np.float64)
    p[1] = 0.0  # floor base
    size = caster["size_m"]
    w = max(size["width_m"], size["depth_m"])

    # 4 corners of an oval directly under the caster.
    sw = max(w * 0.45, 0.10)
    pts_world = [
        p + np.array([+sw, 0.0,  0.0]),
        p + np.array([-sw, 0.0,  0.0]),
        p + np.array([ 0.0, 0.0, +sw * 0.6]),
        p + np.array([ 0.0, 0.0, -sw * 0.6]),
    ]
    screen_pts: list[tuple[float, float]] = []
    for wpt in pts_world:
        proj = _project(wpt, *cam)
        if proj is None:
            return
        screen_pts.append((proj[0], proj[1]))
    _draw_dark_ellipse(buf, screen_pts, _CONTACT_DARKEN)


def _add_cast_shadow(buf: np.ndarray, caster: dict, light_pos: np.ndarray,
                     intensity: float, cam) -> None:
    p = np.array(caster["position_m"], dtype=np.float64)
    p[1] = 0.0
    L = np.asarray(light_pos, dtype=np.float64)
    size = caster["size_m"]
    h_obj = float(size["height_m"])
    w_obj = max(size["width_m"], size["depth_m"])

    d_xz = np.array([p[0] - L[0], 0.0, p[2] - L[2]])
    n_xz = float(np.linalg.norm(d_xz))
    if n_xz < 1e-3:
        return
    d_xz /= n_xz

    light_height = max(L[1] - p[1], 0.10)
    raw_length = h_obj * (n_xz / light_height)
    shadow_length = float(np.clip(raw_length, _MIN_CAST_LENGTH_M,
                                  _MAX_CAST_LENGTH_FACTOR * w_obj))
    if shadow_length < _MIN_CAST_LENGTH_M:
        return

    perp = np.array([-d_xz[2], 0.0, d_xz[0]])
    half_minor = w_obj * 0.40
    base = p
    tip  = p + d_xz * shadow_length
    mid  = p + d_xz * (shadow_length * 0.5)
    pts_world = [base, tip,
                 mid + perp * half_minor,
                 mid - perp * half_minor]

    screen_pts: list[tuple[float, float]] = []
    for wpt in pts_world:
        proj = _project(wpt, *cam)
        if proj is None:
            return
        screen_pts.append((proj[0], proj[1]))

    darken = float(np.clip(_CAST_DARKEN_PER_INTENSITY * intensity, 0.0, 0.55))
    _draw_dark_ellipse(buf, screen_pts, darken)


def _camera_basis(camera: dict, W: int, H: int):
    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"], dtype=np.float64)
    up_w = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    fwd = look_at - cam_pos
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, up_w); right /= max(np.linalg.norm(right), 1e-9)
    up_c  = np.cross(right, fwd); up_c /= max(np.linalg.norm(up_c), 1e-9)
    fx = W / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
    return cam_pos, right, up_c, fwd, fx, W / 2.0, H / 2.0


def _window_directional_lightpos(window_src: dict, room_dims: dict) -> np.ndarray | None:
    """Approximate a window as a 'sun' point-light high above the room on the
    side opposite the wall — so casters inside the room shadow toward the
    centre. Returns world-space (3,) light position."""
    pos = np.array(window_src["position_m"], dtype=np.float64)
    wall = window_src.get("wall")
    width  = float(room_dims.get("width_m",  4.0))
    depth  = float(room_dims.get("depth_m",  4.0))
    height = float(room_dims.get("height_m", 2.6))

    high = max(height, 2.5) + 1.5
    if wall == "back":
        return np.array([pos[0], high, -2.0])
    if wall == "front":
        return np.array([pos[0], high, depth + 2.0])
    if wall == "left":
        return np.array([-2.0, high, pos[2]])
    if wall == "right":
        return np.array([width + 2.0, high, pos[2]])
    return None


def add_shadows(buf: np.ndarray, output_dir: str | Path,
                lights: dict, sources: list[dict], camera: dict) -> None:
    """Apply contact + cast shadows to `buf` (mutated in place)."""
    casters = collect_casters(output_dir)
    if not casters:
        return

    H, W = buf.shape[:2]
    cam = _camera_basis(camera, W, H)

    # Contact shadows under every caster (always)
    for c in casters:
        _add_contact_shadow(buf, c, cam)

    # Cast shadows: active point lights as point sources, daylight windows
    # as elevated directional sources.
    by_id = {s["id"]: s for s in sources if "id" in s}

    # Read room dimensions so window shadows have a sane elevation/offset.
    room_dims: dict = {}
    fp_path = Path(output_dir) / "floorplan_analysis.json"
    if fp_path.exists():
        try:
            room = json.loads(fp_path.read_text()).get("room", {})
            room_dims = room.get("dimensions") or {}
        except Exception:
            pass

    for p in lights.get("point_lights", []):
        if not p.get("on", True) or float(p.get("intensity", 0.0)) <= 1e-3:
            continue
        s = by_id.get(p["id"])
        if s is None:
            continue
        anchor = np.array(s["position_m"], dtype=np.float64)
        offset = np.array(p.get("offset_m", [0, 0.05, 0]), dtype=np.float64)
        light_pos = anchor + offset
        for c in casters:
            _add_cast_shadow(buf, c, light_pos, float(p["intensity"]), cam)

    for w in lights.get("windows", []):
        if not w.get("daylight_on", True) or float(w.get("intensity", 0.0)) <= 1e-3:
            continue
        s = by_id.get(w["id"])
        if s is None:
            continue
        light_pos = _window_directional_lightpos(s, room_dims)
        if light_pos is None:
            continue
        for c in casters:
            _add_cast_shadow(buf, c, light_pos, float(w["intensity"]) * 0.7, cam)
