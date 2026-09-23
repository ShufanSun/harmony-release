"""pyrender-based shadow / lighting render.

This is the "real" renderer: it loads the assembled 3D scene (walls +
furniture + decorations + wall-mounted), sets up the camera to match
camera_vggt.json, attaches lights from `lights.json` as proper
`pyrender.PointLight` / `DirectionalLight` / ambient, and rasterises with
shadow maps enabled. The output is saved to disk; nothing is shown on
screen.

Headless execution
------------------
The renderer uses `pyrender.OffscreenRenderer`, which still needs an
OpenGL context. If the host has neither EGL nor OSMesa, wrap the python
invocation with `xvfb-run`:

    xvfb-run -a python -m lighting_module.pipeline --output-dir <run> ...

Xvfb provides a virtual frame buffer entirely in RAM — no display server
or screen is opened, the rendered image is written straight to PNG.

Falls back gracefully (returns None with a clear log message) if the GL
context can't be created — the 2D `render_lit.py` preview is always
available as a fallback.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# pyrender 0.1.45's shadow backend references np.infty, removed in NumPy 2.0.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]


def _load_camera(out_dir: Path) -> dict | None:
    for cand in (out_dir / "camera_vggt.json", out_dir / "camera.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def _camera_pose(camera: dict) -> np.ndarray:
    """HARMONY camera (position_m, look_at_m, up) → pyrender pose matrix.

    pyrender / OpenGL convention: camera looks at -Z, X right, Y up. So the
    pose matrix's columns are [right, up, -forward, position].
    """
    pos = np.asarray(camera["position_m"], dtype=np.float64)
    look_at = np.asarray(camera["look_at_m"], dtype=np.float64)
    up_w = np.asarray(camera.get("up", [0, 1, 0]), dtype=np.float64)

    fwd = look_at - pos
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, up_w); right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, fwd);   up /= max(np.linalg.norm(up), 1e-9)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -fwd
    pose[:3, 3] = pos
    return pose


def _room_dims(out_dir: Path) -> dict:
    fp = out_dir / "floorplan_analysis.json"
    dims = {"width_m": 4.0, "depth_m": 4.0, "height_m": 2.6}
    if fp.exists():
        try:
            room = json.loads(fp.read_text()).get("room", {})
            # The pipeline schema is room.floor_width_m / floor_depth_m /
            # ceiling_height_m. Earlier prototypes used a nested
            # room.dimensions sub-object — fall back to that for old runs.
            sub = room.get("dimensions", {}) or {}
            dims["width_m"]  = float(
                room.get("floor_width_m")
                or sub.get("width_m") or sub.get("width") or 4.0)
            dims["depth_m"]  = float(
                room.get("floor_depth_m")
                or sub.get("depth_m") or sub.get("depth") or 4.0)
            dims["height_m"] = float(
                room.get("ceiling_height_m")
                or sub.get("height_m") or sub.get("height") or 2.6)
        except Exception:
            pass
    return dims


def _textured_quad_pyrender_mesh(corners_xyz: np.ndarray, tex_path: Path,
                                  uv_tile: tuple[float, float] = (1.0, 1.0)):
    """Build a `pyrender.Mesh` directly (no trimesh visual layer) from 4
    world-space corners (CCW from the visible side) and a texture image.

    Going through `trimesh.SimpleMaterial` + `pyrender.Mesh.from_trimesh` was
    silently dropping the texture in this version of pyrender (room rendered
    flat gray even though the PNGs existed on disk). Constructing the
    pyrender Primitive + MetallicRoughnessMaterial + Texture explicitly is
    more verbose but actually applies the image.

    `uv_tile` controls how many texture repeats span the quad. Default 1×1
    stretches one tile across the full surface (preserves prior behavior);
    pass e.g. (4, 4) to repeat the texture 4×4 times for finer wall detail.
    """
    import pyrender
    from PIL import Image

    verts = corners_xyz.astype(np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uv = np.array([[0.0,         0.0],
                   [uv_tile[0],  0.0],
                   [uv_tile[0],  uv_tile[1]],
                   [0.0,         uv_tile[1]]], dtype=np.float32)

    if tex_path.exists():
        try:
            img = np.array(Image.open(str(tex_path)).convert("RGB"))
            texture = pyrender.Texture(source=img, source_channels="RGB")
            material = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=texture,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=1.0,
                doubleSided=True,
            )
        except Exception as e:
            print(f"[pyrender] texture load failed for {tex_path.name}: {e}")
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.6, 0.6, 0.6, 1.0],
                metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
    else:
        print(f"[pyrender] missing texture {tex_path.name} — using gray fallback")
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.6, 0.6, 0.6, 1.0],
            metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)

    primitive = pyrender.Primitive(
        positions=verts,
        indices=indices.flatten(),
        texcoord_0=uv,
        material=material,
        mode=4,  # GL_TRIANGLES
    )
    return pyrender.Mesh(primitives=[primitive])


def _add_room(scene, out_dir: Path) -> None:
    """Build the room as 6 textured quads (floor, ceiling, 4 walls). This
    replaces walls_with_objects.obj which has no UVs/textures."""
    dims = _room_dims(out_dir)
    w, d, h = dims["width_m"], dims["depth_m"], dims["height_m"]

    quads = [
        # Floor: visible from above (CCW when viewed from +Y)
        ("floor_texture.png",
         np.array([[0, 0, 0], [w, 0, 0], [w, 0, d], [0, 0, d]])),
        # Ceiling: visible from below (CCW when viewed from -Y)
        ("ceiling_texture.png",
         np.array([[0, h, 0], [0, h, d], [w, h, d], [w, h, 0]])),
        # Back wall (Z=0): visible from +Z; CCW from camera side
        ("wall_back_texture.png",
         np.array([[0, 0, 0], [0, h, 0], [w, h, 0], [w, 0, 0]])),
        # Front wall (Z=d): visible from -Z
        ("wall_front_texture.png",
         np.array([[w, 0, d], [w, h, d], [0, h, d], [0, 0, d]])),
        # Left wall (X=0): visible from +X
        ("wall_left_texture.png",
         np.array([[0, 0, d], [0, h, d], [0, h, 0], [0, 0, 0]])),
        # Right wall (X=w): visible from -X
        ("wall_right_texture.png",
         np.array([[w, 0, 0], [w, h, 0], [w, h, d], [w, 0, d]])),
    ]

    for tex_name, corners in quads:
        tex_path = out_dir / tex_name
        if not tex_path.exists():
            # Fallbacks: wall_texture.png (single-wall case), then bare grey
            alt = out_dir / "wall_texture.png"
            tex_path = alt if alt.exists() else tex_path
        mesh = _textured_quad_pyrender_mesh(corners, tex_path)
        scene.add(mesh)


def _add_carpets(scene, out_dir: Path) -> int:
    """Carpets are placed by the furniture stage as flat textured quads
    (4 world-space corners on the floor plane), not as GLBs. Pull them out
    of furniture_placements.json and add them as additional textured quads
    that sit ~5mm above the floor so they don't z-fight."""
    json_path = out_dir / "furniture" / "furniture_placements.json"
    if not json_path.exists():
        return 0
    try:
        entries = json.loads(json_path.read_text())
    except Exception as e:
        print(f"[pyrender] furniture_placements.json read failed: {e}")
        return 0

    n = 0
    for entry in entries:
        if not entry.get("is_carpet"):
            continue
        corners = entry.get("world_corners")
        carpet_img = entry.get("carpet_img")
        if not corners or len(corners) != 4 or not carpet_img:
            continue
        # Lift slightly above floor (Y += 0.005m) to avoid z-fighting
        verts = np.asarray(corners, dtype=np.float64)
        verts[:, 1] += 0.005
        tex_path = Path(carpet_img)
        if not tex_path.exists():
            # The placement JSON sometimes carries an absolute path written
            # at generation time; try the local mirror under this run dir.
            alt = out_dir / "furniture" / "inpainted" / Path(carpet_img).name
            tex_path = alt if alt.exists() else tex_path
        mesh = _textured_quad_pyrender_mesh(verts, tex_path)
        scene.add(mesh)
        n += 1
    return n


def _wall_mounted_pose(entry: dict) -> np.ndarray:
    """Build a 4×4 transform that places a wall-mounted GLB at its
    `world_pt` on the right wall, scaled to `size_m`, oriented to face
    into the room.

    Convention: GLBs from the wall-mounted stage are roughly axis-aligned
    in their local space. We reset the bounds to a unit cube, scale to
    size_m, rotate to the wall's outward face direction, then translate to
    world_pt with a small inward offset of half the depth so the back face
    of the GLB sits flush with the wall.
    """
    wall = entry.get("wall") or "back"
    size = entry.get("size_m") or {}
    sw = float(size.get("width_m",  0.6))
    sh = float(size.get("height_m", 0.9))
    sd = float(size.get("depth_m",  0.1))

    # Wall-direction rotation (GLB local +Z → wall inward normal).
    R_wall = {
        "back":  np.eye(3),
        "left":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
        "right": np.array([[0, 0,  1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
        "front": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    }.get(wall, np.eye(3))

    # Optional extra yaw from glb_front_deg.
    yaw = np.radians(float(entry.get("glb_front_deg", 0.0) or 0.0))
    if abs(yaw) > 1e-3:
        c, s = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        R_wall = R_wall @ Ry

    # Inward normal for the depth offset.
    inward = {
        "back":  np.array([0.0, 0.0,  1.0]),
        "front": np.array([0.0, 0.0, -1.0]),
        "left":  np.array([1.0, 0.0,  0.0]),
        "right": np.array([-1.0, 0.0, 0.0]),
    }.get(wall, np.array([0.0, 0.0, 1.0]))

    pos = np.asarray(entry.get("world_pt", [0, 0, 0]), dtype=np.float64) \
          + inward * (sd * 0.5)

    pose = np.eye(4)
    pose[:3, :3] = R_wall
    pose[:3, 3] = pos
    return pose, np.array([sw, sh, sd], dtype=np.float64)


def _add_wall_mounted(scene, out_dir: Path) -> int:
    """Load each wall-mounted GLB at its placement pose. Keeps the GLB's
    own textures (rectified window/door inpaint, art image, etc.)."""
    import pyrender, trimesh
    json_path = out_dir / "wall_mounted" / "placements" / "object_placements.json"
    if not json_path.exists():
        return 0
    try:
        entries = json.loads(json_path.read_text())
    except Exception as e:
        print(f"[pyrender] wall-mounted json read failed: {e}")
        return 0

    n = 0
    for entry in entries:
        glb_name = entry.get("glb_file") or ""
        is_window = "window" in glb_name.lower()
        glb_path = out_dir / "wall_mounted" / "objects" / glb_name
        if not glb_path.exists():
            continue
        try:
            loaded = trimesh.load(str(glb_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(loaded.dump())
        except Exception as e:
            print(f"[pyrender] {glb_name} load failed: {e}")
            continue

        # Centre and scale the GLB to size_m (the GLBs ship in arbitrary
        # local units depending on how they were generated).
        b = loaded.bounds
        center = (b[0] + b[1]) / 2.0
        extents = np.maximum(b[1] - b[0], 1e-6)
        pose, target_size = _wall_mounted_pose(entry)
        scale_xyz = target_size / extents

        verts = (loaded.vertices.astype(np.float64) - center) * scale_xyz
        loaded = loaded.copy()
        loaded.vertices = verts

        if is_window:
            # Override the window's material with a strong emissive so the
            # pane visibly glows on the room-side view. The lit_*.glb path
            # tries to do this via emissive.py but stores emissiveFactor in
            # uint8 [0,255], which gets clamped/dropped by glTF (must be
            # [0,1]) — set it directly here in pyrender units instead.
            pr_mesh = pyrender.Mesh.from_trimesh(loaded, smooth=False)
            for prim in pr_mesh.primitives:
                prim.material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 0.95, 0.85, 1.0],
                    emissiveFactor=[3.0, 2.85, 2.55],
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    doubleSided=True,
                )
            scene.add(pr_mesh, pose=pose)
        else:
            scene.add(pyrender.Mesh.from_trimesh(loaded, smooth=False), pose=pose)
        n += 1
    return n


def _add_glb(scene, glb_path: Path, pose: np.ndarray | None = None) -> bool:
    import pyrender, trimesh
    if not glb_path.exists():
        return False
    try:
        loaded = trimesh.load(str(glb_path), force="scene")
        meshes = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
        for m in meshes:
            scene.add(pyrender.Mesh.from_trimesh(m, smooth=False),
                      pose=pose if pose is not None else np.eye(4))
        return True
    except Exception as e:
        print(f"[pyrender] GLB load failed for {glb_path.name}: {e}")
        return False


def _attach_lights(scene, lights: dict, sources: list[dict]) -> int:
    """Translate `lights.json` into pyrender lights. Returns count attached."""
    import pyrender

    n = 0
    by_id = {s["id"]: s for s in sources if "id" in s}

    # Ambient: pyrender Scene.ambient_light is a 3-vector RGB intensity.
    # Floor at ~0.15 so shadowed regions don't crush to pitch black — even a
    # high-contrast daylit room has bounce light on the unlit walls.
    amb = lights.get("ambient") or {}
    color = np.asarray(amb.get("color", [1, 1, 1]), dtype=np.float64)
    intensity = float(amb.get("intensity", 0.0))
    ambient_rgb = np.maximum(color * intensity, 0.15)
    scene.ambient_light = ambient_rgb.astype(np.float32)

    # Point lights: lamps + ceiling fixtures. pyrender 0.1.45's shadow
    # backend doesn't implement point-light shadows, so we model each lamp
    # as a wide-cone SpotLight pointing downward — this is a fair physical
    # approximation (a real lampshade emits primarily through its open
    # bottom) and gives proper shadow maps.
    for p in lights.get("point_lights", []):
        if not p.get("on", True) or float(p.get("intensity", 0.0)) <= 1e-3:
            continue
        s = by_id.get(p["id"])
        if s is None:
            continue
        anchor = np.asarray(s["position_m"], dtype=np.float64)
        offset = np.asarray(p.get("offset_m", [0.0, 0.05, 0.0]), dtype=np.float64)

        # Spot pose: emitter looking down -Y. pyrender emits along -Z of the
        # local pose, so rotate -Z into world -Y by mapping y_world ↦ -z_local.
        pose = np.eye(4)
        pose[:3, 0] = [1.0, 0.0,  0.0]   # local +X = world +X
        pose[:3, 1] = [0.0, 0.0,  1.0]   # local +Y = world +Z
        pose[:3, 2] = [0.0, 1.0,  0.0]   # local +Z = world +Y → -Z = -Y (down)
        pose[:3, 3] = anchor + offset

        watts = float(p["intensity"]) * 220.0
        light = pyrender.SpotLight(
            color=np.asarray(p.get("color", [1, 1, 1]), dtype=np.float32),
            intensity=watts,
            innerConeAngle=np.pi / 6.0,    # 30° fully-lit core
            outerConeAngle=np.pi / 2.4,    # 75° falloff
        )
        scene.add(light, pose=pose)
        n += 1

    # Directional lights from explicit entries + windows treated as 'sun'.
    def _add_dir(dir_world: np.ndarray, color, intensity_unit: float):
        nonlocal n
        if intensity_unit <= 1e-3:
            return
        d = np.asarray(dir_world, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            return
        d /= norm
        # pyrender DirectionalLight emits along -Z of its pose. Build a
        # pose whose -Z column equals the world-space light direction.
        z = -d
        if abs(z[1]) < 0.9:
            up_ref = np.array([0.0, 1.0, 0.0])
        else:
            up_ref = np.array([1.0, 0.0, 0.0])
        x = np.cross(up_ref, z); x /= max(np.linalg.norm(x), 1e-9)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        # Position the directional 5m back along its negative direction
        # so its frustum encompasses the room.
        pose[:3, 3] = -d * 5.0
        # Directional intensity calibration: pyrender's DirectionalLight
        # uses radiance-like units. Daylight through a window in this scene
        # graph wants ~10-15 to show a clear bright→dark gradient across the
        # room without blowing out the lit faces. Was 6.0 (felt like dusk).
        light = pyrender.DirectionalLight(
            color=np.asarray(color, dtype=np.float32),
            intensity=float(intensity_unit) * 12.0,
        )
        scene.add(light, pose=pose)
        n += 1

    for d in lights.get("directional", []):
        _add_dir(d.get("direction", [0, -1, 0]),
                 d.get("color", [1, 1, 1]),
                 float(d.get("intensity", 0.0)))

    for w in lights.get("windows", []):
        if not w.get("daylight_on", True):
            continue
        s = by_id.get(w["id"])
        if s is None:
            continue
        wall = s.get("wall")
        # Sun streams INTO the room (away from the wall the window is on).
        if wall == "back":
            direction = np.array([0.0, -0.4, 1.0])
        elif wall == "front":
            direction = np.array([0.0, -0.4, -1.0])
        elif wall == "left":
            direction = np.array([1.0, -0.4, 0.0])
        elif wall == "right":
            direction = np.array([-1.0, -0.4, 0.0])
        else:
            direction = np.array([0.0, -1.0, 0.0])
        _add_dir(direction, w.get("color", [1, 1, 1]), float(w.get("intensity", 0.0)))

    return n


def render(output_dir: str | Path, lights: dict, sources: list[dict],
           dst: Path) -> Path | None:
    """Render the assembled scene with pyrender + shadow maps. Returns dst on
    success, or None if the GL context could not be created."""
    out_dir = Path(output_dir)

    # Default to EGL when nothing is set; user will wrap with xvfb-run if
    # neither EGL nor OSMesa is available.
    if "PYOPENGL_PLATFORM" not in os.environ and not os.environ.get("DISPLAY"):
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    try:
        import pyrender
    except Exception as e:
        print(f"[pyrender] import failed ({e}); skipping shadow render.")
        return None

    camera_dict = _load_camera(out_dir)
    if camera_dict is None:
        print("[pyrender] no camera_vggt.json — skipping")
        return None

    W = int(camera_dict.get("width_px", 1920))
    H = int(camera_dict.get("height_px", 1080))
    # Cap viewport size so the offscreen render stays under a few hundred MB.
    if W > 1600:
        scale = 1600.0 / W
        W = int(W * scale); H = int(H * scale)

    scene = pyrender.Scene(ambient_light=np.array([0.0, 0.0, 0.0]),
                           bg_color=np.array([20.0, 20.0, 22.0]))

    # ── Room geometry: textured floor / ceiling / 4 walls ──
    _add_room(scene, out_dir)

    # ── Carpets: flat textured quads from furniture_placements.json ──
    n_carpet = _add_carpets(scene, out_dir)
    if n_carpet:
        print(f"[pyrender] attached {n_carpet} carpet(s)")

    # ── Wall-mounted objects (windows, mirrors, art) at world poses ──
    n_wm = _add_wall_mounted(scene, out_dir)
    print(f"[pyrender] attached {n_wm} wall-mounted object(s)")

    # ── Furniture + decorations (pre-assembled with textures) ──
    scene_glb_loaded = False
    for cand in (out_dir / "decorations" / "placements" / "scene_with_decoration.glb",
                 out_dir / "furniture" / "scene_with_furniture.glb"):
        if cand.exists():
            scene_glb_loaded = _add_glb(scene, cand)
            if scene_glb_loaded:
                break
    if not scene_glb_loaded:
        print("[pyrender] no furniture/decoration scene GLB found — room will be empty")

    # ── Camera ──
    hfov_rad = np.radians(float(camera_dict["hfov_deg"]))
    yfov = 2.0 * float(np.arctan(np.tan(hfov_rad / 2.0) * H / W))
    cam = pyrender.PerspectiveCamera(yfov=yfov, znear=0.05, zfar=50.0)
    scene.add(cam, pose=_camera_pose(camera_dict))

    # ── Lights ──
    n_lights = _attach_lights(scene, lights, sources)
    print(f"[pyrender] attached {n_lights} light(s)  viewport={W}x{H}")

    # ── Render ──
    try:
        r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    except Exception as e:
        print(f"[pyrender] OffscreenRenderer failed ({e}). "
              f"On a headless host: wrap your command with `xvfb-run -a`.")
        return None

    try:
        # pyrender 0.1.45: SHADOWS_POINT raises "not implemented". Use
        # directional + spot only (lamps were attached as SpotLights above).
        flags = (pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
                 | pyrender.constants.RenderFlags.SHADOWS_SPOT)
        color, depth = r.render(scene, flags=flags)
    except Exception as e:
        print(f"[pyrender] render failed ({e})")
        r.delete()
        return None

    r.delete()

    from PIL import Image
    Image.fromarray(color).save(str(dst))
    print(f"[pyrender] saved → {dst}")
    return dst
