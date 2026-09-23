"""
place_ceiling_objects.py — hang reconstructed ceiling fixtures from the ceiling
plane and composite them into the scene render.

Counterpart to the wall-mounted / furniture placement stages, specialised for
CEILING geometry — which is far simpler than wall placement: a fixture hangs
straight down from the ceiling, so its placement is fully determined by

  1. the (x, z) where the bounding-box centre pixel ray pierces the ceiling
     plane  y = ceiling_height,  and
  2. a physical size inferred from the mask's pixel extent at that depth.

The fixture GLB is scaled to that size and translated so the TOP of the mesh
(the canopy / chain attachment) meets the ceiling, then it is composited with
pyrender (textured + lit, matching the furniture compositor) on top of the
latest scene render.

Reads <output_dir>/ceiling/segment_results.json (each segment must already have
a ``glb_file`` from object_generation), the calibrated camera, and the
floorplan room dimensions.

Usage:
    python -m object_placement.ceiling.place_ceiling_objects \\
        --output-dir outputs/front3d/rgb_003200 \\
        --image outputs/front3d/rgb_003200/ref.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes

# ── Plausible ceiling-fixture size envelopes (metres) ─────────────────────────
#   width  = horizontal extent;  drop = vertical hang below the ceiling.
_SIZE_BOUNDS: dict[str, dict] = {
    "pendant_light": dict(w=(0.12, 0.70), drop=(0.15, 1.10), w_def=0.30, drop_def=0.40),
    "chandelier":    dict(w=(0.35, 1.40), drop=(0.25, 1.20), w_def=0.70, drop_def=0.55),
    "ceiling_lamp":  dict(w=(0.20, 0.90), drop=(0.05, 0.30), w_def=0.45, drop_def=0.15),
    "ceiling_fan":   dict(w=(0.80, 1.60), drop=(0.25, 0.60), w_def=1.20, drop_def=0.35),
    "track_light":   dict(w=(0.40, 1.60), drop=(0.05, 0.25), w_def=0.90, drop_def=0.12),
    "other":         dict(w=(0.15, 1.00), drop=(0.10, 0.80), w_def=0.40, drop_def=0.40),
}


# ── Scene I/O ─────────────────────────────────────────────────────────────────

def _load_camera(out_root: Path) -> dict:
    camera_p = (out_root / "camera_vggt.json"
                if (out_root / "camera_vggt.json").exists()
                else out_root / "camera.json")
    if not camera_p.exists():
        raise FileNotFoundError(f"Camera file not found: {camera_p}")
    with open(camera_p) as f:
        return json.load(f)


def _load_room(out_root: Path) -> tuple[float, float, float]:
    """Return (room_w, room_d, ceiling_h).

    Room footprint comes from walls.obj — the SAME authoritative geometry the
    furniture/seating pipeline uses — so the ceiling 'room centre' matches the
    rendered room.  floorplan_analysis.json's floor_width/depth_m are frequently
    stale or wrong (003454: 2.64×5.0 vs the real 4.67×4.36), which put the 'room
    centre' off to the side and off-frame.  Ceiling height still comes from the
    floorplan analysis (walls.obj doesn't carry it reliably)."""
    room_w, room_d, ceiling_h = 5.0, 5.0, 2.7
    fp_path = out_root / "floorplan_analysis.json"
    if fp_path.exists():
        try:
            with open(fp_path) as f:
                room = json.load(f).get("room", {})
            room_w   = float(room.get("floor_width_m",  room_w)) or room_w
            room_d   = float(room.get("floor_depth_m",  room_d)) or room_d
            ceiling_h = float(room.get("ceiling_height_m", ceiling_h)) or ceiling_h
        except Exception as e:
            print(f"[ceil_place] could not read floorplan_analysis.json ({e}) — defaults")
    walls_p = out_root / "walls.obj"
    if walls_p.exists():
        try:
            import numpy as _np
            _v = _np.array([[float(x) for x in l.split()[1:4]]
                            for l in open(walls_p) if l.startswith("v ")])
            _ow = float(_v[:, 0].max() - _v[:, 0].min())
            _od = float(_v[:, 2].max() - _v[:, 2].min())
            if _ow > 0.5 and _od > 0.5:
                if abs(_ow - room_w) > 0.3 or abs(_od - room_d) > 0.3:
                    print(f"[ceil_place] room dims from walls.obj → {_ow:.2f}×{_od:.2f} "
                          f"(floorplan_analysis said {room_w:.2f}×{room_d:.2f})")
                room_w, room_d = _ow, _od
        except Exception as e:
            print(f"[ceil_place] could not read walls.obj dims ({e}) — using floorplan analysis")
    return room_w, room_d, ceiling_h


# ── Camera projection helpers (match the furniture pyrender compositor) ───────

def _camera_basis(cam: dict):
    pos     = np.array(cam["position_m"], dtype=np.float64)
    look_at = np.array(cam["look_at_m"],  dtype=np.float64)
    up_w    = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    W, H    = int(cam["width_px"]), int(cam["height_px"])
    hfov    = float(cam["hfov_deg"])
    right, up_c, fwd = _camera_axes(pos, look_at, up_w)
    fx = W / (2.0 * math.tan(math.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0
    return pos, right, up_c, fwd, fx, cx, cy, W, H


def _ray_for_pixel(px: float, py: float, right, up_c, fwd, fx, cx, cy) -> np.ndarray:
    """Unit world ray through image pixel (px, py)."""
    d = right * ((px - cx) / fx) + up_c * ((cy - py) / fx) + fwd
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


def _backproject_to_plane_y(pos, ray, plane_y: float) -> np.ndarray | None:
    """Intersect a world ray with the horizontal plane y = plane_y."""
    if abs(ray[1]) < 1e-6:
        return None
    t = (plane_y - pos[1]) / ray[1]
    if t <= 0.01:
        return None
    return pos + t * ray


def _estimate_size(box_px, anchor, pos, fwd, fx, obj_type: str) -> tuple[float, float]:
    """Pinhole size at the fixture depth, clamped to a plausible envelope."""
    b = _SIZE_BOUNDS.get(obj_type, _SIZE_BOUNDS["other"])
    zc = float(np.dot(anchor - pos, fwd))               # depth along view axis
    if zc <= 0.05:
        return b["w_def"], b["drop_def"]
    bw_px = max(1.0, box_px[2] - box_px[0])
    bh_px = max(1.0, box_px[3] - box_px[1])
    width_m = (bw_px / fx) * zc
    drop_m  = (bh_px / fx) * zc
    width_m = float(np.clip(width_m, *b["w"]))
    drop_m  = float(np.clip(drop_m,  *b["drop"]))
    return width_m, drop_m


# ── Placement ─────────────────────────────────────────────────────────────────

def _build_placement(seg: dict, ceil_dir: Path, cam: dict,
                     room_w: float, room_d: float, ceiling_h: float,
                     center_it: bool = True) -> dict | None:
    """Compute scale + world transform for one ceiling fixture.

    Returns a placement dict (glb_path, scale, rotation_3x3, position_m, anchor
    metadata) compatible with the pyrender compositor below, or None if the
    fixture has no GLB / cannot be projected onto the ceiling.
    """
    import trimesh

    obj_type = seg.get("type", "other")
    glb_rel  = seg.get("glb_file")
    if not glb_rel:
        print(f"  [{obj_type}] no glb_file — skip")
        return None
    glb_path = (ceil_dir / glb_rel) if not Path(glb_rel).is_absolute() else Path(glb_rel)
    if not glb_path.exists():
        print(f"  [{obj_type}] GLB missing: {glb_path} — skip")
        return None

    box = seg.get("box_px")
    if not box:
        print(f"  [{obj_type}] no box_px — skip")
        return None

    pos, right, up_c, fwd, fx, cx, cy, W, H = _camera_basis(cam)

    # Bounding-box centre → ray → ceiling plane.
    bcx = (box[0] + box[2]) / 2.0
    bcy = (box[1] + box[3]) / 2.0
    ray = _ray_for_pixel(bcx, bcy, right, up_c, fwd, fx, cx, cy)
    anchor = _backproject_to_plane_y(pos, ray, ceiling_h)
    if anchor is None:
        # The centre-pixel ray does not rise to the ceiling plane — the box sits
        # below the horizon, so this is NOT a real ceiling fixture (a GDINO false
        # positive on a table lamp, the sofa, glare, etc.).  Drop it: a genuine
        # ceiling fixture's box centre always projects up onto the ceiling.  This
        # acts as a geometric backstop when the VLM context-verify is unavailable.
        print(f"  [{obj_type}] box centre does not project onto the ceiling "
              f"(below horizon) — dropping as non-ceiling false positive")
        return None
    # Keep the anchor inside the room footprint.
    anchor[0] = float(np.clip(anchor[0], 0.1, room_w - 0.1))
    anchor[2] = float(np.clip(anchor[2], 0.1, room_d - 0.1))

    # A hanging ceiling fixture (chandelier / pendant / ceiling lamp / flush mount)
    # hangs near the CENTRE of the room, but nudged toward where its silhouette
    # actually appears. The bbox back-projection (`anchor`) is depth-ambiguous for
    # a ceiling object and on its own often lands in a corner (003025:
    # (0.31,0.10)); the room centre on its own ignores the ref. So blend: start at
    # the room centre and shift toward the silhouette back-projection, bounded so
    # the fixture stays central and never reaches a wall/corner.
    _LIGHT_T = ("chandelier", "pendant", "ceiling_lamp", "ceiling_light",
                "flush_mount", "dome_light", "hanging", "lamp", "light")
    if any(k in obj_type.lower() for k in _LIGHT_T):
        _silh_x, _silh_z = float(anchor[0]), float(anchor[2])   # silhouette back-projection
        # Place at the SILHOUETTE (where the reference shows the fixture in image
        # space), then pull the DEPTH toward the CAMERA. Back-projecting the
        # fixture's bbox centre onto the ceiling plane overshoots AWAY from the
        # camera — the body hangs below the ceiling, so a mid-fixture pixel maps to
        # a too-far ceiling point, which jams the fixture into the back wall (003454)
        # or pulls a foreground chandelier backward (004124). User's rule: use the
        # silhouette to place, then bring it toward the camera by depth.
        _ax, _az = _silh_x, _silh_z
        _to = np.array([float(pos[0]) - _ax, float(pos[2]) - _az])
        _dist = float(np.linalg.norm(_to))
        if _dist > 1e-3:
            _pull = min(0.40 * _dist, 1.3)
            _ax += float(_to[0]) / _dist * _pull
            _az += float(_to[1]) / _dist * _pull
        if center_it:
            anchor[0] = float(np.clip(_ax, 0.12 * room_w, 0.88 * room_w))
            anchor[2] = float(np.clip(_az, 0.12 * room_d, 0.88 * room_d))
            print(f"  [{obj_type}] MAIN light → silhouette + pulled toward camera "
                  f"→ ({anchor[0]:.2f},{anchor[2]:.2f})  (silh ({_silh_x:.2f},{_silh_z:.2f}))")
        else:
            anchor[0] = float(np.clip(_ax, 0.18 * room_w, 0.82 * room_w))
            anchor[2] = float(np.clip(_az, 0.18 * room_d, 0.82 * room_d))
            print(f"  [{obj_type}] secondary light → silhouette + pulled toward camera "
                  f"→ ({anchor[0]:.2f},{anchor[2]:.2f})  (silh ({_silh_x:.2f},{_silh_z:.2f}))")

    width_m, drop_m = _estimate_size(box, anchor, pos, fwd, fx, obj_type)

    # Load mesh, derive uniform scale from horizontal extent (preserves the
    # fixture's own proportions; the mask only sets overall size).
    try:
        loaded = trimesh.load(str(glb_path), force="scene")
        subs = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        mesh = max(subs, key=lambda g: len(g.faces))
    except Exception as e:
        print(f"  [{obj_type}] mesh load failed: {e} — skip")
        return None

    ext = mesh.extents  # (ex, ey, ez)
    horiz = float(max(ext[0], ext[2])) or 1.0
    s = width_m / horiz
    # Don't let the resulting vertical drop exceed the plausible cap.
    b = _SIZE_BOUNDS.get(obj_type, _SIZE_BOUNDS["other"])
    if ext[1] * s > b["drop"][1]:
        s = b["drop"][1] / (float(ext[1]) or 1.0)
    scale = [s, s, s]

    print(f"  [{obj_type}] anchor=({anchor[0]:.2f},{ceiling_h:.2f},{anchor[2]:.2f}) "
          f"w={width_m:.2f}m drop≈{ext[1]*s:.2f}m scale×{s:.3f}")

    return {
        "index":        seg.get("index"),
        "type":         obj_type,
        "glb_path":     str(glb_path),
        "scale":        scale,
        "rotation_3x3": np.eye(3).tolist(),
        "position_m":   [float(anchor[0]), float(ceiling_h), float(anchor[2])],
        "ceiling_h":    float(ceiling_h),
    }


def _ceiling_transform(mesh, scale: np.ndarray, R: np.ndarray,
                       pos: np.ndarray) -> np.ndarray:
    """Scale + rotate the mesh, then translate so its TOP meets the ceiling and
    its horizontal centroid sits at (pos.x, pos.z).  The fixture hangs DOWN from
    pos.y (= ceiling height)."""
    verts = mesh.vertices.astype(np.float64).copy()
    # Centre XZ, drop to local origin at top.
    cx = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    cz = (verts[:, 2].min() + verts[:, 2].max()) / 2.0
    verts[:, 0] -= cx
    verts[:, 2] -= cz
    verts = verts * scale[np.newaxis, :]
    verts = verts @ R.T
    # Anchor the TOP of the mesh to y = pos[1] (ceiling), hang downward.
    top = float(verts[:, 1].max())
    verts[:, 1] += (pos[1] - top)
    # Re-centre XZ after rotation and translate to the anchor XZ.
    rcx = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    rcz = (verts[:, 2].min() + verts[:, 2].max()) / 2.0
    verts[:, 0] += pos[0] - rcx
    verts[:, 2] += pos[2] - rcz
    return verts


# ── Render / composite (textured pyrender, matched to furniture compositor) ───

def render_ceiling_pyrender(output_dir: Path, camera: dict, placements: list[dict],
                            base_image_path: Path | None = None,
                            out_path: Path | None = None,
                            render_scale: int = 2) -> Path | None:
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender  # noqa: F401
    except Exception as e:
        print(f"[ceil_render] pyrender unavailable ({e}) — skipping composite")
        return None
    import trimesh
    from PIL import Image
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    out_root = Path(output_dir)
    ceil_dir = out_root / "ceiling"

    # Base background: prefer the most complete scene render available so the
    # pendant composites over the furnished room.
    if base_image_path is not None:
        base_path = Path(base_image_path)
    else:
        cands = [
            # Prefer the most complete furnished render so the ceiling fixture
            # composites on TOP of furniture + decorations.  The decoration
            # stage runs after ceiling, so its render is the newest; layering
            # the pendant over it yields a final image with everything.
            out_root / "decorations" / "placements" / "render_decorations_placed.png",
            out_root / "furniture" / "render_furniture_placed.png",
            out_root / "wall_mounted" / "placements" / "render_objects_placed_ref_texture.png",
            out_root / "wall_mounted" / "placements" / "render_objects_placed.png",
            out_root / "render_ref_texture.png",
            out_root / "render_final.png",
            out_root / "render.png",
        ]
        base_path = next((p for p in cands if p.exists()), None)
    if base_path is None or not Path(base_path).exists():
        print("[ceil_render] no base render found — skipping")
        return None
    print(f"[ceil_render] Base render: {Path(base_path).name}")

    base_img = Image.open(base_path).convert("RGB")
    W0, H0 = base_img.size
    W, H = W0 * render_scale, H0 * render_scale
    if render_scale != 1:
        base_img = base_img.resize((W, H), Image.LANCZOS)
    base_arr = np.array(base_img, dtype=np.uint8)

    pos, right, up_c, fwd, _fx, _cx, _cy, _W, _H = _camera_basis(camera)
    hfov = float(camera["hfov_deg"])
    fx = W / (2.0 * math.tan(math.radians(hfov / 2.0)))
    yfov = 2.0 * math.atan((H / 2.0) / fx)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up_c
    pose[:3, 2] = -fwd
    pose[:3, 3] = pos

    pr_scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                              ambient_light=[0.9, 0.9, 0.9])
    n_added = 0
    for p in placements:
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        try:
            loaded = trimesh.load(str(gp), force="scene")
            subs = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
            mesh = max(subs, key=lambda g: len(g.faces))
        except Exception as e:
            print(f"[ceil_render] load failed {gp.name}: {e}")
            continue

        scale = np.array(p["scale"],        dtype=np.float64)
        R     = np.array(p["rotation_3x3"], dtype=np.float64)
        posv  = np.array(p["position_m"],   dtype=np.float64)
        verts = _ceiling_transform(mesh, scale, R, posv)

        # Brighten the dark baked albedo (same approach as the furniture
        # compositor — keep the GLB's own UV atlas, just lift the tone).
        visual = mesh.visual
        try:
            mat = getattr(mesh.visual, "material", None)
            tex = getattr(mat, "baseColorTexture", None) if mat is not None else None
            if tex is not None and getattr(mesh.visual, "uv", None) is not None:
                tarr = np.asarray(tex, dtype=np.float32)
                m = float(tarr[..., :3].mean()) or 1.0
                gain = float(np.clip(120.0 / m, 1.0, 2.2))
                if gain > 1.01:
                    tarr[..., :3] = np.clip(tarr[..., :3] * gain, 0, 255)
                    visual = TextureVisuals(
                        uv=np.array(mesh.visual.uv),
                        material=PBRMaterial(
                            baseColorTexture=Image.fromarray(tarr.astype(np.uint8)),
                            metallicFactor=0.0, roughnessFactor=1.0),
                    )
        except Exception:
            visual = mesh.visual

        placed = trimesh.Trimesh(vertices=verts, faces=mesh.faces,
                                 visual=visual, process=False)
        try:
            pr_scene.add(pyrender.Mesh.from_trimesh(placed, smooth=False), pose=np.eye(4))
            n_added += 1
        except Exception as e:
            print(f"[ceil_render] skip {p.get('type')}: {e}")

    if n_added == 0:
        print("[ceil_render] no renderable ceiling geometry — skipping")
        return None

    pr_scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H,
                                            znear=0.05, zfar=100.0), pose=pose)
    pr_scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5), pose=pose)

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    try:
        color, depth = r.render(pr_scene)
    finally:
        r.delete()

    # ── Occluder depth pass: correct z-buffering ────────────────────────────
    # The fixture above was rendered ALONE, so `depth>0` pastes it on top of the
    # photo everywhere it projects — even where a wall/furniture is in front of
    # it. Render the room shell + furniture (depth only) and drop fixture pixels
    # that sit *behind* that geometry, so the pendant is properly occluded. The
    # photo base is kept as the background (real windows etc.), so this gives
    # correct z-buffer AND correct background.
    occ_depth = None
    try:
        from lighting_module.render_pyrender import _add_glb, _add_room, _add_wall_mounted
        occ = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.0, 0.0, 0.0])
        _add_room(occ, out_root)
        _add_wall_mounted(occ, out_root)
        for cand in (out_root / "decorations" / "placements" / "scene_with_decoration.glb",
                     out_root / "furniture" / "scene_with_furniture.glb"):
            if cand.exists() and _add_glb(occ, cand):
                break
        occ.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H,
                                           znear=0.05, zfar=100.0), pose=pose)
        ro = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        try:
            occ_depth = ro.render(occ, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
        finally:
            ro.delete()
    except Exception as e:
        print(f"[ceil_render] occluder pass failed ({e}); compositing without z-test")
        occ_depth = None

    mask = depth > 0
    if occ_depth is not None:
        occluded = (occ_depth > 0) & (occ_depth < depth - 2e-3)
        n_occ = int((mask & occluded).sum())
        mask = mask & ~occluded
        if n_occ:
            print(f"[ceil_render] z-buffer: hid {n_occ} occluded fixture px")
    out = base_arr.copy()
    out[mask] = color[mask, :3]

    if out_path is None:
        out_path = ceil_dir / "render_ceiling_placed.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(out_path))
    print(f"[ceil_render] textured composite → {out_path}  "
          f"({n_added} geoms, {W}×{H}, {int(mask.sum())} fixture px)")
    return out_path


def _save_scene_glb(placements: list[dict], out_path: Path) -> None:
    """Save the placed fixtures (world-space) as a combined GLB for downstream
    use (lighting / full-scene assembly)."""
    import trimesh
    scene = trimesh.Scene()
    for p in placements:
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        try:
            loaded = trimesh.load(str(gp), force="scene")
            subs = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
            mesh = max(subs, key=lambda g: len(g.faces))
            verts = _ceiling_transform(mesh, np.array(p["scale"], dtype=np.float64),
                                       np.array(p["rotation_3x3"], dtype=np.float64),
                                       np.array(p["position_m"], dtype=np.float64))
            scene.add_geometry(trimesh.Trimesh(vertices=verts, faces=mesh.faces,
                                               visual=mesh.visual, process=False),
                               geom_name=f"ceiling_{p.get('index')}_{p.get('type')}")
        except Exception as e:
            print(f"[ceil_place] scene-glb skip {gp.name}: {e}")
    if len(scene.geometry):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(out_path))
        print(f"[ceil_place] scene GLB → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(output_dir: str | Path, image_path: str | Path | None = None,
        use_vlm: bool = True, base_image_path: str | Path | None = None) -> Path:
    out_root = Path(output_dir)
    ceil_dir = out_root / "ceiling"
    results_p = ceil_dir / "segment_results.json"
    if not results_p.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_ceiling_objects first.\n"
            f"Expected: {results_p}"
        )

    cam = _load_camera(out_root)
    room_w, room_d, ceiling_h = _load_room(out_root)
    print(f"[ceil_place] room {room_w:.1f}×{room_d:.1f} m, ceiling {ceiling_h:.2f} m")

    with open(results_p) as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        print("[ceil_place] No segments — nothing to place.")
        return ceil_dir

    placements: list[dict] = []
    # The single (or LARGEST) ceiling fixture is the room's main/central light →
    # centre it; any additional fixtures are secondary and kept at their own
    # silhouette position (a room can have a central chandelier + a side pendant).
    def _box_area(s):
        b = s.get("box_px") or [0, 0, 0, 0]
        return (b[2] - b[0]) * (b[3] - b[1])
    _main_idx = max(range(len(segments)), key=lambda i: _box_area(segments[i])) if segments else -1
    for _si, seg in enumerate(segments):
        _is_main = (_si == _main_idx)
        print(f"[ceil_place] {seg.get('index'):02d} {seg.get('type')}"
              f"{' (main → centre)' if _is_main else ' (secondary → silhouette)'}")
        pl = _build_placement(seg, ceil_dir, cam, room_w, room_d, ceiling_h, center_it=_is_main)
        if pl is not None:
            placements.append(pl)
            seg["placement"] = {
                "position_m":   pl["position_m"],
                "scale":        pl["scale"],
                "rotation_3x3": pl["rotation_3x3"],
            }

    if not placements:
        print("[ceil_place] No fixtures could be placed.")
        with open(results_p, "w") as f:
            json.dump(data, f, indent=2)
        return ceil_dir

    # Persist placements back into the results JSON.
    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    _save_scene_glb(placements, ceil_dir / "scene_with_ceiling.glb")
    render_ceiling_pyrender(out_root, cam, placements,
                            base_image_path=Path(base_image_path) if base_image_path else None)
    print(f"[ceil_place] placed {len(placements)} ceiling fixture(s) → {ceil_dir}")
    return ceil_dir


def main():
    ap = argparse.ArgumentParser(description="Place ceiling fixtures hanging from the ceiling.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--base-render", default=None)
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    run(output_dir=args.output_dir, image_path=args.image,
        use_vlm=not args.no_vlm, base_image_path=args.base_render)


if __name__ == "__main__":
    main()
