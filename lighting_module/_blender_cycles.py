"""_blender_cycles.py — runs INSIDE Blender (bpy), not the scenegen env.

Invoked head-less by render_blender.py:

    blender --background --python lighting_module/_blender_cycles.py -- <config.json>

It imports the assembled scene GLB, places physical Cycles lights at the
VLM-estimated light sources (windows → daylight area lights, lamps/sconces →
warm point lights, ceiling fixtures → downward area lights), sets the camera
from camera.json, and path-traces a single frame with AgX tonemapping + GI.

`config.json` schema (written by render_blender.py):
  {
    "scene_glb": "...scene_with_decoration.glb",
    "extra_objects": [{"glb": "...track_light.glb",
                       "position_m":[x,y,z], "scale":[s,s,s]}],
    "camera": {"position_m":[...], "look_at_m":[...], "up":[...],
               "hfov_deg":70.0},
    "room_center_m": [x,y,z],
    "lights": [{"kind":"window|lamp|ceiling_fixture",
                "position_m":[x,y,z], "color":[r,g,b], "power":1200.0,
                "size":[w,h]}],
    "world": {"color":[r,g,b], "strength":0.3},
    "resolution": [625, 350], "samples": 160,
    "view_transform": "AgX", "exposure": 0.0,
    "output_png": "...render_blender.png"
  }
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

# glTF (Y-up) → Blender (Z-up) basis change, as a 4×4. The Blender glTF importer
# applies this same conversion to the scene, so camera/lights placed through it
# land in the same frame.  (x, y, z) → (x, -z, y).
_C4 = np.array([[1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]], dtype=float)


def g2b(p):
    """glTF / HARMONY (Y-up) → Blender (Z-up): (x, y, z) → (x, -z, y)."""
    return Vector((p[0], -p[2], p[1]))


def _camera_matrix(cam):
    """Exact pyrender camera-to-world pose (OpenGL: looks -Z, +Y up, +X right),
    converted gltf→Blender — Blender cameras share the -Z look convention, so
    this reproduces the pyrender view 1:1."""
    pos = np.asarray(cam["position_m"], float)
    look = np.asarray(cam["look_at_m"], float)
    up_w = np.asarray(cam.get("up", [0, 1, 0]), float)
    fwd = look - pos
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, up_w); right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, fwd);   up /= max(np.linalg.norm(up), 1e-9)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -fwd
    pose[:3, 3] = pos
    M = _C4 @ pose
    return Matrix([list(M[i]) for i in range(4)])


def _aim_euler(direction):
    """Euler so the object's local -Z points along `direction` (Blender)."""
    d = Vector(direction)
    if d.length < 1e-9:
        d = Vector((0, 0, -1))
    return d.normalized().to_track_quat('-Z', 'Y').to_euler()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    cfg = json.load(open(argv[0]))

    # --- clean factory scene ---
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # --- Cycles + device ---
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = int(cfg.get("samples", 160))
    scene.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        chosen = 'CPU'
        for dev_type in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == dev_type]
                if gpus:
                    for d in prefs.devices:
                        d.use = (d.type == dev_type)
                    chosen = 'GPU'
                    break
            except Exception:
                continue
        scene.cycles.device = chosen
        print(f"[blender] Cycles device = {chosen}")
    except Exception as e:
        scene.cycles.device = 'CPU'
        print(f"[blender] device setup failed ({e}); CPU")

    # --- output / colour management ---
    res = cfg.get("resolution", [625, 350])
    scene.render.resolution_x = int(res[0])
    scene.render.resolution_y = int(res[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = cfg.get("view_transform", "AgX")
    except Exception:
        scene.view_settings.view_transform = "Filmic"
    scene.view_settings.exposure = float(cfg.get("exposure", 0.0))

    # --- world ambient (so shadowed regions aren't pure black) ---
    _white = bool(cfg.get("light_white"))
    w = cfg.get("world", {"color": [0.05, 0.06, 0.08], "strength": 0.3})
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        wcol = w["color"]
        if _white:  # neutral grey ambient (keep brightness, drop the tint)
            g = sum(wcol) / 3.0
            wcol = [g, g, g]
        bg.inputs[0].default_value = (*wcol, 1.0)
        bg.inputs[1].default_value = float(w["strength"])

    # --- import the assembled scene ---
    bpy.ops.import_scene.gltf(filepath=cfg["scene_glb"])
    print(f"[blender] imported scene: {cfg['scene_glb']}")

    # --- wire vertex colors into Base Color so ColorVisuals objects don't render
    #     flat white. trimesh bakes some objects (curtains, and any furniture/wall
    #     object whose texture couldn't be preserved through geo_save) as per-vertex
    #     colors (glTF COLOR_0). Blender's glTF import stores these as a mesh color
    #     attribute but does NOT connect them to the shader, so the object shows the
    #     material's (white) Base Color. For any mesh that has a color attribute but
    #     whose Base Color has no texture input, connect a Color-Attribute node to
    #     Base Color. Textured objects (Base Color already linked) are left alone.
    #     Disable with SCENEWEAVE_NO_VCOL=1.
    if os.environ.get("SCENEWEAVE_NO_VCOL") != "1":
        _vcol_fixed = 0
        for o in bpy.data.objects:
            if o.type != 'MESH':
                continue
            cattrs = getattr(o.data, "color_attributes", None)
            if not cattrs or len(cattrs) == 0:
                continue
            cname = cattrs[0].name
            for slot in getattr(o, "material_slots", []):
                mat = slot.material
                if not mat or not mat.use_nodes:
                    continue
                nt = mat.node_tree
                bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf is None:
                    continue
                bc = bsdf.inputs.get('Base Color')
                if bc is None or bc.is_linked:
                    continue   # already textured — leave it
                try:
                    vc = nt.nodes.new('ShaderNodeVertexColor')
                    vc.layer_name = cname
                    nt.links.new(vc.outputs['Color'], bc)
                except Exception:
                    va = nt.nodes.new('ShaderNodeAttribute')
                    va.attribute_name = cname
                    nt.links.new(va.outputs['Color'], bc)
                _vcol_fixed += 1
        print(f"[blender] wired vertex colors into base color on "
              f"{_vcol_fixed} material slot(s)")

    # --- backface-cull the room shell so an OUTSIDE camera sees INTO the room ---
    # VGGT legitimately places the camera outside the scene box for some scenes; the
    # outer wall between camera and room would then occlude the whole view (a flat
    # blown-out wall close-up). The shell is built with inward-facing normals, so
    # making BACKFACES transparent hides only the outer side of each surface —
    # interior (inside-camera) views are unchanged, but from outside the near wall
    # turns see-through and the room interior renders. Disable with
    # SCENEWEAVE_NO_SHELL_CULL=1.
    if os.environ.get("SCENEWEAVE_NO_SHELL_CULL") != "1":
        _SHELL_TOKENS = ("wall_back", "wall_left", "wall_right", "wall_front",
                         "ceiling", "floor")

        def _is_shell(name):
            n = (name or "").lower()
            return any(t in n for t in _SHELL_TOKENS) and "wallobj" not in n

        def _backface_cull(mat):
            if not mat or not mat.use_nodes:
                return
            nt = mat.node_tree
            out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if out is None:
                return
            surf = out.inputs.get('Surface')
            if surf is None or not surf.is_linked:
                return
            src = surf.links[0].from_socket
            geo = nt.nodes.new('ShaderNodeNewGeometry')
            transp = nt.nodes.new('ShaderNodeBsdfTransparent')
            mix = nt.nodes.new('ShaderNodeMixShader')
            nt.links.new(geo.outputs['Backfacing'], mix.inputs['Fac'])  # 0 front / 1 back
            nt.links.new(src, mix.inputs[1])                            # front -> original
            nt.links.new(transp.outputs['BSDF'], mix.inputs[2])         # back  -> transparent
            nt.links.new(mix.outputs['Shader'], surf)

        _culled = 0
        for o in bpy.data.objects:
            if o.type == 'MESH' and _is_shell(o.name):
                for slot in getattr(o, "material_slots", []):
                    _backface_cull(slot.material)
                    _culled += 1
        print(f"[blender] shell backface-cull applied to {_culled} material slot(s)")

    # --- import extra objects (ceiling fixture) at their placement ---
    # glTF import of the emissive lit_ copies leaves 'Emission Strength' at 0
    # (only 'Emission Color' is set from emissiveFactor), so the fixture never
    # glows.  Force an absolute strength on the "emissive" material here — the
    # emissive_scale multiply pass below can't help (0 × scale = 0).
    _fixture_emis = float(os.environ.get("SCENEWEAVE_FIXTURE_EMISSION", "12.0"))
    _extra_object_names = set()
    for ob in cfg.get("extra_objects", []):
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=ob["glb"])
        except Exception as e:
            print(f"[blender] extra import failed {ob['glb']}: {e}")
            continue
        news = [o for o in bpy.data.objects if o not in before]
        _extra_object_names.update(o.name for o in news)
        roots = [o for o in news if o.parent is None]
        s = ob.get("scale", [1, 1, 1])
        pos = g2b(ob["position_m"])
        for r in roots:
            r.scale = (r.scale[0] * s[0], r.scale[1] * s[1], r.scale[2] * s[2])
            r.location = r.location + pos
        # light the emissive fixture faces
        for o in news:
            for slot in getattr(o, "material_slots", []):
                mat = slot.material
                if not mat or "emissive" not in mat.name.lower() or not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        es = node.inputs.get('Emission Strength')
                        ec = node.inputs.get('Emission Color')
                        if es is not None and es.default_value < _fixture_emis:
                            es.default_value = _fixture_emis
                        if ec is not None:
                            if _white:                       # neutral white glow
                                ec.default_value = (1.0, 1.0, 1.0, 1.0)
                            elif sum(ec.default_value[:3]) < 0.05:
                                ec.default_value = (1.0, 0.82, 0.45, 1.0)
                    elif node.type == 'EMISSION':
                        st = node.inputs.get('Strength')
                        if st is not None and st.default_value < _fixture_emis:
                            st.default_value = _fixture_emis
        print(f"[blender] placed extra: {ob['glb']}")

    # --- re-apply glass transmission stripped by the trimesh assembly ---
    # The per-object GLB carries KHR_materials_transmission, but baking it into
    # scene_full.glb drops the extension, so glass tables / acrylic chairs come
    # in opaque.  Match each transmissive object's placed position to the
    # nearest imported mesh and set Transmission on its Principled BSDF.
    glass = cfg.get("glass_objects", [])
    if glass:
        # Extras (carpet, ceiling fixture, …) are placed independently of
        # furniture_placements/decoration_placements — a flat rug's bbox
        # centroid can sit closer to a table's floor-level placement anchor
        # than the table's own (elevated) centroid, so exclude them here.
        meshes = [o for o in bpy.data.objects
                  if o.type == 'MESH' and o.name not in _extra_object_names]

        def _centroid(o):
            corners = [o.matrix_world @ Vector(c) for c in o.bound_box]
            acc = Vector((0.0, 0.0, 0.0))
            for c in corners:
                acc += c
            return acc / 8.0

        cents = [(o, _centroid(o)) for o in meshes]
        for gobj in glass:
            target = g2b(gobj["position_m"])
            best, bestd = None, 1e9
            for o, c in cents:
                d = (c - target).length
                if d < bestd:
                    best, bestd = o, d
            if best is None or bestd > 1.0:
                print(f"[blender] glass: no mesh within 1m of {gobj['position_m']} "
                      f"(nearest {bestd:.2f}m) — skip")
                continue
            # Tinted glass: KEEP the object's own base colour/texture (do NOT
            # disconnect it or force white — that strips the colour). Just add
            # transmission so it reads as COLOURED glass. Cap transmission so the
            # colour stays clearly visible rather than washing out to clear.
            trans = min(0.7, float(gobj["transmission"]))
            rough = min(0.05, float(gobj["roughness"]))
            for slot in best.material_slots:
                mat = slot.material
                if not mat or not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type != 'BSDF_PRINCIPLED':
                        continue
                    mt = node.inputs.get('Metallic')
                    if mt is not None:
                        mt.default_value = 0.0
                    tw = (node.inputs.get('Transmission Weight')
                          or node.inputs.get('Transmission'))
                    if tw is not None:
                        tw.default_value = trans
                    rg = node.inputs.get('Roughness')
                    if rg is not None:
                        rg.default_value = rough
                    ip = node.inputs.get('IOR')
                    if ip is not None:
                        ip.default_value = float(gobj["ior"])
            print(f"[blender] glass: clear (transmission {trans}) → "
                  f"'{best.name}' ({bestd:.2f}m from placement)")

    # --- dim self-lit (emissive) surfaces, e.g. blown-out window glass ---
    emis_scale = float(cfg.get("emissive_scale", 1.0))
    if emis_scale != 1.0:
        n = 0
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    es = node.inputs.get('Emission Strength')
                    if es is not None and es.default_value:
                        es.default_value = float(es.default_value) * emis_scale; n += 1
                elif node.type == 'EMISSION':
                    st = node.inputs.get('Strength')
                    if st is not None and st.default_value:
                        st.default_value = float(st.default_value) * emis_scale; n += 1
        print(f"[blender] scaled emission ×{emis_scale} on {n} material slot(s)")

    # --- camera ---
    cam_cfg = cfg["camera"]
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.angle = math.radians(float(cam_cfg.get("hfov_deg", 60.0)))
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.matrix_world = _camera_matrix(cam_cfg)
    scene.camera = cam_obj

    # --- lights ---
    center = g2b(cfg.get("room_center_m", [0, 1, 0]))
    for i, L in enumerate(cfg.get("lights", [])):
        kind = L.get("kind", "lamp")
        pos = g2b(L["position_m"])
        color = L.get("color", [1.0, 1.0, 1.0])
        power = float(L.get("power", 100.0))
        if kind == "window":
            ld = bpy.data.lights.new(f"win_{i}", 'AREA')
            ld.shape = 'RECTANGLE'
            sz = L.get("size", [0.8, 2.0])
            ld.size, ld.size_y = sz[0], sz[1]
        elif kind == "ceiling_fixture":
            ld = bpy.data.lights.new(f"ceil_{i}", 'AREA')
            ld.shape = 'DISK'
            ld.size = L.get("size", [0.4])[0]
        else:  # lamp / sconce
            ld = bpy.data.lights.new(f"lamp_{i}", 'POINT')
            ld.shadow_soft_size = 0.15
        ld.color = color
        ld.energy = power
        lo = bpy.data.objects.new(ld.name, ld)
        scene.collection.objects.link(lo)
        # nudge area lights slightly off the wall/ceiling into the room so they
        # don't clip the surface they sit on, and aim them inward / downward.
        if kind == "window":
            aim = (center - pos); aim.z = min(aim.z, -0.15)  # bias slightly down
            lo.location = pos + aim.normalized() * 0.12
            lo.rotation_euler = _aim_euler(center - pos)
        elif kind == "ceiling_fixture":
            lo.location = pos - Vector((0, 0, 0.05))
            lo.rotation_euler = _aim_euler(Vector((0, 0, -1)))
        else:
            lo.location = pos
        print(f"[blender] light {kind} @ {tuple(round(x,2) for x in pos)} "
              f"power={power} color={[round(c,2) for c in color]}")

    # --- render ---
    scene.render.filepath = cfg["output_png"]
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {cfg['output_png']}")


if __name__ == "__main__":
    main()
