"""
assemble_scene_glb.py — merge the whole reconstructed scene into ONE textured GLB.

The pipeline leaves the scene split across several files:
  - walls.obj                         room shell (geometry only)
  - walls_metadata[_ref].json         per-surface reference textures + UV info
  - furniture/scene_with_furniture.glb
  - ceiling/scene_with_ceiling.glb
  - wall_mounted/.../scene_with_*.glb (if any)

This script builds a textured room shell (back / left / right walls + floor +
ceiling, each as a quad with its reference texture and planar UVs) and appends
every placed-object scene GLB (already in world coordinates), then exports a
single ``scene_full.glb`` that opens as a complete, viewable scene.

Usage:
    python object_placement/assemble_scene_glb.py --output-dir outputs/front3d/rgb_003266
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


def _load_tex(path: Path, fallback=(200, 200, 200)) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (8, 8), fallback)


def _quad(corners: list[np.ndarray], uv: np.ndarray, tex: Image.Image,
          name: str, inward_to: np.ndarray | None = None) -> trimesh.Trimesh:
    """Two-triangle quad with a baseColor texture and per-vertex UVs.

    corners: 4 world points in winding order [v0, v1, v2, v3].
    uv:      (4, 2) UV coords matching corners.
    inward_to: if given (room centre), the face winding is flipped when needed
        so the surface normal points TOWARD this point.  Room surfaces must face
        the interior, otherwise they are back-face-culled (and render as dark /
        missing) when viewed from inside the room.
    """
    verts = np.array(corners, dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    if inward_to is not None:
        e1 = verts[1] - verts[0]
        e2 = verts[2] - verts[0]
        n = np.cross(e1, e2)
        face_c = verts.mean(axis=0)
        if float(np.dot(np.asarray(inward_to, float) - face_c, n)) < 0:
            faces = faces[:, ::-1].copy()   # flip winding → normal points inward

    # Double-sided so the surface is visible from both sides regardless of the
    # viewer's position (belt-and-suspenders alongside the inward winding).
    mat = PBRMaterial(baseColorTexture=tex, metallicFactor=0.0,
                      roughnessFactor=1.0, doubleSided=True)
    visual = TextureVisuals(uv=uv.astype(np.float64), material=mat)
    m = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
    m.metadata["name"] = name
    return m


def build_room_shell(out_root: Path) -> list[trimesh.Trimesh]:
    """Construct textured floor / ceiling / back / left / right quads."""
    meta_p = (out_root / "walls_metadata_ref.json"
              if (out_root / "walls_metadata_ref.json").exists()
              else out_root / "walls_metadata.json")
    walls_obj = out_root / "walls.obj"
    if not walls_obj.exists():
        print(f"[assemble] walls.obj missing ({walls_obj}) — no room shell")
        return []

    # Room extent from the wall mesh bounds (authoritative geometry).
    box = trimesh.load(str(walls_obj), force="mesh")
    lo, hi = box.bounds
    W = float(hi[0] - lo[0])      # x extent (room width)
    Hh = float(hi[1] - lo[1])     # y extent (ceiling height)
    Dd = float(hi[2] - lo[2])     # z extent (room depth)
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    print(f"[assemble] room {W:.2f}×{Hh:.2f}×{Dd:.2f} m  (x∈[{x0:.2f},{x1:.2f}] "
          f"y∈[{y0:.2f},{y1:.2f}] z∈[{z0:.2f},{z1:.2f}])")

    meta: dict = {}
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
        except Exception as e:
            print(f"[assemble] could not parse {meta_p.name}: {e}")

    def _tex_for(orient: str, default_name: str) -> Image.Image:
        entry = meta.get(orient, {}) if isinstance(meta, dict) else {}
        tp = entry.get("texture_path")
        if tp:
            p = Path(tp)
            if not p.is_absolute():
                p = (out_root / p) if not str(p).startswith("outputs") else Path(p)
            if not p.exists():
                p = out_root / Path(tp).name
            if p.exists():
                return _load_tex(p)
        # fallbacks by conventional filename
        for cand in (out_root / f"{default_name}_ref.png", out_root / f"{default_name}.png"):
            if cand.exists():
                return _load_tex(cand)
        return _load_tex(Path("/nonexistent"))

    def _tile(orient: str, default: float) -> float:
        e = meta.get(orient, {}) if isinstance(meta, dict) else {}
        try:
            return float(e.get("tile_size_m", default)) or default
        except Exception:
            return default

    shells: list[trimesh.Trimesh] = []
    ctr = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])  # room centre

    # All quads are oriented inward (normal → room centre) via `inward_to`, so
    # the literal corner winding below only needs to be a valid quad loop.

    # ── Floor (y = y0); UV maps (x, z) ───────────────────────────────────────
    t = _tile("floor", W)
    shells.append(_quad(
        [np.array([x0, y0, z0]), np.array([x1, y0, z0]),
         np.array([x1, y0, z1]), np.array([x0, y0, z1])],
        np.array([[0, 0], [W / t, 0], [W / t, Dd / t], [0, Dd / t]]),
        _tex_for("floor", "floor_texture"), "floor", inward_to=ctr))

    # ── Ceiling (y = y1); UV maps (x, z) ─────────────────────────────────────
    t = _tile("ceiling", 2.0)
    shells.append(_quad(
        [np.array([x0, y1, z0]), np.array([x0, y1, z1]),
         np.array([x1, y1, z1]), np.array([x1, y1, z0])],
        np.array([[0, 0], [0, Dd / t], [W / t, Dd / t], [W / t, 0]]),
        _tex_for("ceiling", "ceiling_texture"), "ceiling", inward_to=ctr))

    # ── Back wall (z = z0); UV maps (x, y) ───────────────────────────────────
    t = _tile("back", W)
    shells.append(_quad(
        [np.array([x0, y0, z0]), np.array([x1, y0, z0]),
         np.array([x1, y1, z0]), np.array([x0, y1, z0])],
        np.array([[0, 1], [W / t, 1], [W / t, 0], [0, 0]]),
        _tex_for("back", "wall_back_texture"), "wall_back", inward_to=ctr))

    # ── Left wall (x = x0); UV maps (z, y) ───────────────────────────────────
    t = _tile("left", Dd)
    shells.append(_quad(
        [np.array([x0, y0, z0]), np.array([x0, y0, z1]),
         np.array([x0, y1, z1]), np.array([x0, y1, z0])],
        np.array([[0, 1], [Dd / t, 1], [Dd / t, 0], [0, 0]]),
        _tex_for("left", "wall_left_texture"), "wall_left", inward_to=ctr))

    # ── Right wall (x = x1); UV maps (z, y) ──────────────────────────────────
    t = _tile("right", Dd)
    shells.append(_quad(
        [np.array([x1, y0, z0]), np.array([x1, y1, z0]),
         np.array([x1, y1, z1]), np.array([x1, y0, z1])],
        np.array([[0, 1], [0, 0], [Dd / t, 0], [Dd / t, 1]]),
        _tex_for("right", "wall_right_texture"), "wall_right", inward_to=ctr))

    return shells


def _find_object_scenes(out_root: Path) -> list[Path]:
    """All placed-object scene GLBs (furniture / ceiling / wall-mounted).

    If the decoration stage has run it exports ``scene_with_decoration.glb``,
    which ALREADY contains the furniture merged with the decor objects — so we
    prefer it over ``scene_with_furniture.glb`` to avoid double-counting the
    furniture.
    """
    deco_scene = out_root / "decorations" / "placements" / "scene_with_decoration.glb"
    furn_scene = (deco_scene if deco_scene.exists()
                  else out_root / "furniture" / "scene_with_furniture.glb")
    cands = [
        furn_scene,
        out_root / "ceiling"   / "scene_with_ceiling.glb",
    ]
    # wall-mounted scene GLB has varied names — glob for it.
    cands += sorted((out_root / "wall_mounted").rglob("scene_with_*.glb"))
    cands += sorted((out_root / "wall_mounted").rglob("placements/*scene*.glb"))
    seen, out = set(), []
    for p in cands:
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _wall_objects_textured(out_root: Path) -> list[trimesh.Trimesh]:
    """Rebuild the placed wall-mounted objects as TEXTURED meshes.

    ``walls_with_objects.obj`` stores each object's world-space geometry (one
    ``# segment`` block per object) but with no UVs/material — the wall-mounted
    placement only writes vertices/faces.  However ``_transform_glb_to_wall``
    applies only per-vertex affine transforms (centre, flips, scale, OBB
    rotation, translate), so the baked verts line up 1:1 (same count + order)
    with the source GLB's vertices.  We therefore re-attach each source GLB's
    UVs + material to its baked world-space verts.

    Returns a list of per-object Trimeshes (textured where the vertex count
    matches the source GLB, neutral-grey fallback otherwise)."""
    import re
    place_dir = out_root / "wall_mounted" / "placements"
    objp = place_dir / "walls_with_objects.obj"
    if not objp.exists():
        return []

    # segment_index → (glb_file, wall) from the placement record
    seg_glb: dict[int, str] = {}
    seg_wall: dict[int, str] = {}
    plj = place_dir / "object_placements.json"
    if plj.exists():
        try:
            for p in json.loads(plj.read_text()):
                seg_glb[p.get("segment_index")] = p.get("glb_file")
                seg_wall[p.get("segment_index")] = p.get("wall")
        except Exception as e:
            print(f"[assemble] could not parse object_placements.json: {e}")

    # Parse the OBJ: global vertex list + per-segment face blocks (verts for a
    # segment are a contiguous run, written in source-GLB order).
    gverts: list[tuple] = []
    segs: list[dict] = []
    cur: dict | None = None
    for line in objp.read_text().splitlines():
        if line.startswith("v "):
            p = line.split()
            gverts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("# segment"):
            m = re.match(r"# segment (\d+) \((.+)\)", line)
            cur = {"idx": int(m.group(1)) if m else -1,
                   "type": (m.group(2) if m else "obj"), "faces": []}
            segs.append(cur)
        elif cur is not None and line.startswith("f "):
            tri = [int(t.split("/")[0]) for t in line.split()[1:]]
            if len(tri) >= 3:
                cur["faces"].append(tri[:3])
    if not gverts or not segs:
        return []
    gverts_np = np.array(gverts, dtype=np.float64)

    glb_cache: dict[str, trimesh.Trimesh | None] = {}
    def _load_glb(gf: str):
        if gf in glb_cache:
            return glb_cache[gf]
        gp = out_root / "wall_mounted" / "objects" / gf
        m = None
        if gp.exists():
            try:
                sc = trimesh.load(str(gp), force="scene")
                gg = list(sc.geometry.values())
                m = (trimesh.util.concatenate(gg) if len(gg) > 1 else gg[0]) if gg else None
            except Exception as e:
                print(f"[assemble] wall-obj GLB load failed {gf}: {e}")
        glb_cache[gf] = m
        return m

    def _planar_uv(verts: np.ndarray, wall: str | None) -> np.ndarray:
        """Planar UV for a flat wall object: U = wall-tangent axis, V = world-Y
        (flipped, image is top-down).  Exact for a rectangular window/art plane
        when the per-vertex pairing isn't available."""
        y = verts[:, 1]
        h = verts[:, 2] if wall in ("left", "right") else verts[:, 0]
        hr = max(float(h.max() - h.min()), 1e-6)
        yr = max(float(y.max() - y.min()), 1e-6)
        u = (h - h.min()) / hr
        v = (y - y.min()) / yr
        return np.column_stack([u, 1.0 - v]).astype(np.float64)

    out: list[trimesh.Trimesh] = []
    n_tex = n_planar = n_grey = 0
    for s in segs:
        if not s["faces"]:
            continue
        faces = np.array(s["faces"], dtype=np.int64)
        vmin, vmax = int(faces.min()), int(faces.max())
        seg_v = gverts_np[vmin - 1:vmax]                 # OBJ is 1-indexed
        lf = faces - vmin                                # 0-indexed local faces
        nseg = vmax - vmin + 1
        gf = seg_glb.get(s["idx"])
        glb = _load_glb(gf) if gf else None
        uv = getattr(getattr(glb, "visual", None), "uv", None) if glb is not None else None
        mat = getattr(getattr(glb, "visual", None), "material", None) if glb is not None else None
        has_tex = mat is not None and getattr(mat, "baseColorTexture", None) is not None
        if (glb is not None and uv is not None and mat is not None
                and len(glb.vertices) == nseg and len(uv) == nseg):
            # exact: re-attach source UVs to the baked world-space verts
            visual = TextureVisuals(uv=np.asarray(uv, np.float64), material=mat)
            mesh = trimesh.Trimesh(vertices=seg_v, faces=lf, visual=visual, process=False)
            n_tex += 1
        elif has_tex:
            # vertex count mismatch (e.g. rebuilt window plane) — planar-project
            # the source texture onto the flat object using its wall axis.
            visual = TextureVisuals(uv=_planar_uv(seg_v, seg_wall.get(s["idx"])),
                                    material=mat)
            mesh = trimesh.Trimesh(vertices=seg_v, faces=lf, visual=visual, process=False)
            n_planar += 1
        else:
            mesh = trimesh.Trimesh(vertices=seg_v, faces=lf, process=False)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=[185, 185, 185, 255])
            n_grey += 1
        mesh.metadata["name"] = f"wallobj_{s['idx']:02d}_{s['type']}"
        out.append(mesh)
    print(f"[assemble] wall-mounted objects: {n_tex} textured (exact UV), "
          f"{n_planar} textured (planar UV), {n_grey} grey-fallback")
    return out


def run(output_dir: str | Path, include_shell: bool = True,
        out_name: str = "scene_full.glb") -> Path:
    out_root = Path(output_dir)
    scene = trimesh.Scene()

    # Load object scenes first so we can tell whether one of them already carries
    # the FULL textured room (a ``walls_with_objects`` geometry: floor + ceiling +
    # 4 walls + wall-mounted objects, baked in by the furniture/decoration stage).
    # If so, building the shell would lay a second floor/ceiling/wall on the exact
    # same planes → z-fighting that flickers and bleeds just past the box edge.
    obj_scenes = []          # (path, {geom_name: mesh})
    for gp in _find_object_scenes(out_root):
        try:
            loaded = trimesh.load(str(gp), force="scene")
        except Exception as e:
            print(f"[assemble] skip {gp.name}: {e}")
            continue
        geoms = (loaded.geometry if isinstance(loaded, trimesh.Scene)
                 else {"g": loaded})
        obj_scenes.append((gp, geoms))

    # Always build the textured room shell.  ``walls_with_objects.obj`` (baked
    # into the furniture/decoration scene GLB) is RAW geometry — no UVs/material
    # — so its walls render untextured.  We DROP that flat mesh below and let the
    # shell provide the textured walls/floor/ceiling, then re-add just the
    # wall-mounted OBJECTS (untextured for now) so they don't disappear.
    n_shell = 0
    if include_shell:
        for m in build_room_shell(out_root):
            scene.add_geometry(m, geom_name=str(m.metadata.get("name", "shell")))
            n_shell += 1

    n_obj = 0
    for gp, geoms in obj_scenes:
        tag = gp.parent.name
        for k, g in geoms.items():
            if include_shell and "walls_with_objects" in k.lower():
                # flat untextured walls+objects — the shell replaces the walls,
                # and _wall_objects_only() re-adds the objects below.
                print(f"[assemble] - dropping flat {k} (replaced by textured shell)")
                continue
            scene.add_geometry(g, geom_name=f"{tag}_{k}")
            n_obj += 1
        print(f"[assemble] + {gp.relative_to(out_root)}  ({len(geoms)} geom)")

    if include_shell:
        for wo in _wall_objects_textured(out_root):
            scene.add_geometry(wo, geom_name=str(wo.metadata.get("name", "wallobj")))
            n_obj += 1

    if not len(scene.geometry):
        raise RuntimeError("nothing to assemble — no shell and no object scenes found")

    out_path = out_root / out_name
    scene.export(str(out_path))
    print(f"\n[assemble] {n_shell} shell + {n_obj} object geom(s) → {out_path}  "
          f"({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


def retexture_scene(output_dir: str | Path, scene_glb: str | Path,
                    out_glb: str | Path | None = None) -> Path:
    """Re-texture an EXISTING object-scene GLB (e.g. ``scene_with_decoration.glb``
    or ``scene_with_ceiling.glb``) without re-running any pipeline stage.

    Drops the raw ``walls_with_objects`` mesh, adds the textured room shell
    (walls/floor/ceiling) and the textured wall-mounted objects, and keeps every
    other (already-textured) geometry.  Overwrites the input GLB unless
    ``out_glb`` is given (a ``.bak_untex`` backup is made on overwrite)."""
    out_root = Path(output_dir)
    scene_glb = Path(scene_glb)
    loaded = trimesh.load(str(scene_glb), force="scene")
    geoms = (loaded.geometry if isinstance(loaded, trimesh.Scene)
             else {"g": loaded})

    scene = trimesh.Scene()
    n_shell = n_obj = 0
    for m in build_room_shell(out_root):
        scene.add_geometry(m, geom_name=str(m.metadata.get("name", "shell")))
        n_shell += 1
    # Idempotent: drop the flat baked walls AND any shell/wall-object geometry
    # already present (e.g. when the input was itself produced by a previous
    # retexture — scene_with_furniture.glb is now pre-retextured), so re-adding
    # the fresh shell + wall objects below doesn't DUPLICATE them (floor_1,
    # wallobj_07_light_1, …).  Non-shell objects (furniture, decorations) pass through.
    _SHELL_PREFIXES = ("floor", "ceiling", "wall_back", "wall_left",
                       "wall_right", "wallobj")
    for k, g in geoms.items():
        kl = k.lower()
        if "walls_with_objects" in kl:
            print(f"[retexture] - dropping flat {k} (replaced by textured shell)")
            continue
        if kl.startswith(_SHELL_PREFIXES):
            continue  # stale shell/wall-object from a prior retexture — re-added fresh
        scene.add_geometry(g, geom_name=k)
        n_obj += 1
    for wo in _wall_objects_textured(out_root):
        scene.add_geometry(wo, geom_name=str(wo.metadata.get("name", "wallobj")))
        n_obj += 1

    out = Path(out_glb) if out_glb else scene_glb
    if out == scene_glb:
        bak = scene_glb.with_suffix(scene_glb.suffix + ".bak_untex")
        if not bak.exists():
            bak.write_bytes(scene_glb.read_bytes())
            print(f"[retexture] backup → {bak.name}")
    scene.export(str(out))
    print(f"[retexture] {n_shell} shell + {n_obj} object geom(s) → {out}  "
          f"({out.stat().st_size/1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Merge the scene into one textured GLB.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-shell", action="store_true",
                    help="Objects only — skip the textured room shell.")
    ap.add_argument("--out-name", default="scene_full.glb")
    ap.add_argument("--retexture",
                    help="Re-texture an existing scene GLB in place (shell + "
                         "wall objects); pass a path relative to --output-dir.")
    args = ap.parse_args()
    if args.retexture:
        retexture_scene(args.output_dir,
                        Path(args.output_dir) / args.retexture)
        return
    run(output_dir=args.output_dir, include_shell=not args.no_shell,
        out_name=args.out_name)


if __name__ == "__main__":
    main()
