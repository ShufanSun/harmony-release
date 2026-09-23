"""render_blender.py — physically-based (Cycles) lighting render.

A path-traced alternative to render_pyrender: imports the assembled scene GLB,
places real Cycles lights at the VLM-estimated light sources (windows →
daylight area lights, lamps/sconces → warm point lights, ceiling fixtures →
downward area lights), and renders with global illumination + AgX tonemapping.
This gives the soft bounce light, rolled-off window highlights, and contact
shadows that pyrender (direct-only, no GI, no tonemap) cannot.

Runs Blender head-less via subprocess (Blender bundles its own Python, so it is
independent of the scenegen env):

    from lighting_module import render_blender
    render_blender.render(out_dir, sources=collected_sources)

Set BLENDER_BIN to override the Blender binary path.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

# Radiometric power (Watts) at the source kind's *nominal* VLM intensity; the
# actual power scales with the VLM-estimated intensity (see _power).
# window default is per-window; scenes with many windows (e.g. 4) over-light and
# blow out — override with SCENEWEAVE_WINDOW_POWER to dim daylight per scene.
_KIND_POWER = {"window": float(os.environ.get("SCENEWEAVE_WINDOW_POWER", 1300.0)),
               "ceiling_fixture": float(os.environ.get("SCENEWEAVE_CEILING_POWER", 250.0)),
               # Room lamps are often the dominant *warm* source in lamp-lit
               # interiors but were fixed at 70 W (vs window 1300) so they could
               # never carry the room. Tunable so the lit scene can be lamp-led.
               "lamp": float(os.environ.get("SCENEWEAVE_LAMP_POWER", 70.0))}
# The VLM intensity considered "normal" for each kind (intensities are small,
# clamped ≤0.5 upstream). Power = base * clamp(intensity / nominal).
_KIND_NOMINAL = {"window": 0.20, "ceiling_fixture": 0.15, "lamp": 0.15}
# Fallback colours (tone) when the VLM has no estimate for a source.
_KIND_COLOR = {
    "window": [0.9, 0.93, 1.0],            # cool daylight
    "ceiling_fixture": [1.0, 0.92, 0.82],  # ~3500K neutral-warm
    "lamp": [1.0, 0.85, 0.65],             # ~2700K warm
}


def _power(kind: str, intensity, scale: float) -> float:
    """Map a VLM intensity (small, ~0.05–0.5) to Cycles Watts for `kind`."""
    base = _KIND_POWER.get(kind, 80.0)
    nom = _KIND_NOMINAL.get(kind, 0.15)
    if intensity is None:
        mult = 1.0
    else:
        # Floor kept low (0.1) so a "too_bright" VLM verdict can actually dim a
        # source — a 0.4 floor previously capped window reduction at 720W and
        # silently overrode the VLM's call to darken them.
        mult = max(0.03, min(float(intensity) / max(nom, 1e-3), 2.5))
    return base * mult * scale


def _find_blender() -> str:
    if os.environ.get("BLENDER_BIN"):
        return os.environ["BLENDER_BIN"]
    for pat in (os.path.expanduser("~/blender/blender-*/blender"),
                "/opt/blender*/blender", "/usr/local/blender*/blender"):
        m = sorted(glob.glob(pat))
        if m:
            return m[-1]
    if shutil.which("blender"):
        return "blender"
    raise FileNotFoundError(
        "Blender not found — set BLENDER_BIN or install to ~/blender/")


def _world_from_ambient(ambient, base_dim: float = 0.5) -> dict:
    """VLM ambient {color,intensity} + base_dim → world background (fill tone).

    base_dim is the VLM's overall ambient-fill level (what render_pyrender uses
    as its baseline); when the explicit sources (windows/sun/fixtures) are all
    judged off, base_dim is the ONLY light the room has, so it must drive the
    world strength — otherwise Cycles renders black while the pyrender preview
    looks correctly lit.
    """
    col = [0.45, 0.45, 0.45]
    inten = float(base_dim)
    if isinstance(ambient, dict):
        if ambient.get("color"):
            col = [max(0.04, x * 0.6) for x in ambient["color"]]
        inten = max(inten, float(ambient.get("intensity", 0.0)) * 1.6)
    strength = max(0.25, min(inten, 1.3))
    # Daylit rooms with no collected window/sun source (all lamps VLM-off) end up
    # lit only by this world fill; the 1.3 clamp is then too dim.  Allow an
    # explicit override to push the world to a daylight level.
    ovr = os.environ.get("SCENEWEAVE_WORLD_STRENGTH")
    if ovr:
        try:
            strength = float(ovr)
        except ValueError:
            pass
    return {"color": col, "strength": strength}


def _room_center(scene_glb: Path) -> list[float]:
    try:
        import trimesh
        s = trimesh.load(str(scene_glb))
        b = s.bounds
        c = (b[0] + b[1]) / 2.0
        return [float(c[0]), float(c[1]), float(c[2])]
    except Exception:
        return [0.0, 1.2, 0.0]


def _collect_glass_objects(out: Path) -> list[dict]:
    """Furniture/decoration glass tables, acrylic chairs etc. carry
    KHR_materials_transmission in their per-object GLB, but the trimesh
    assembly that bakes scene_with_furniture / scene_full STRIPS that
    extension — so the assembled scene renders them opaque.  Re-collect the
    transmissive objects (material_properties.json) with their placed world
    position so Blender can re-apply transmission to the nearest mesh.
    (Ceiling fixtures keep their glass — they're imported separately.)"""
    glass: list[dict] = []
    specs = [
        (out / "furniture" / "material_properties.json",
         out / "furniture" / "furniture_placements.json"),
        (out / "decorations" / "material_properties.json",
         out / "decorations" / "placements" / "decoration_placements.json"),
    ]
    for mat_p, place_p in specs:
        if not mat_p.exists() or not place_p.exists():
            continue
        try:
            mats = json.loads(mat_p.read_text())
            places = json.loads(place_p.read_text())
        except Exception:
            continue
        # Only treat genuinely see-through objects (clear glass/acrylic) this
        # way. A low transmission (ceramic vase, wood cabinet with a stray
        # value) must NOT be forced to clear white — that strips its colour.
        # Seating is never clear glass. Windows/wall-mounted glass are interior
        # wall-objects (no real opening behind them) — clear glass just shows the
        # wall and they vanish, so keep their inpainted daylight-pane texture
        # OPAQUE instead.
        tmap = {k: v for k, v in mats.items()
                if float(v.get("transmission", 0.0) or 0.0) >= 0.6
                and not any(w in k.lower() for w in
                            ("chair", "sofa", "couch", "bed", "stool", "bench",
                             "armchair", "window"))}
        if not tmap:
            continue
        plist = places if isinstance(places, list) else places.get("placements", [])
        for p in plist:
            gp = p.get("glb_path") or p.get("glb") or ""
            props = tmap.get(os.path.basename(gp))
            if props and p.get("position_m"):
                glass.append({
                    "position_m": list(map(float, p["position_m"])),
                    "transmission": float(props.get("transmission", 0.8)),
                    "roughness": float(props.get("roughness", 0.05)),
                    "ior": float(props.get("ior", 1.5)),
                })
    return glass


def render(out_dir: str | Path, sources: list[dict] | None = None,
           output_path: str | Path | None = None, samples: int = 160,
           exposure: float = 0.0, power_scale: float = 1.0) -> str:
    out = Path(out_dir)
    ldir = out / "lightings"
    ldir.mkdir(exist_ok=True)

    # Full assembled scene (shell + wall-mounted + furniture + decorations).
    # Prefer scene_full.glb — the assemble_scene output that includes the
    # textured shell (walls/floor/ceiling). The decorations/placements copy is
    # only furniture+decorations (no shell) and can be a stale intermediate, so
    # it renders objects floating in a grey void.
    scene_glb = out / "scene_full.glb"
    if not scene_glb.exists():
        scene_glb = out / "decorations" / "placements" / "scene_with_decoration.glb"
    if not scene_glb.exists():
        scene_glb = out / "furniture" / "scene_with_furniture.glb"
    if not scene_glb.exists():
        raise FileNotFoundError(f"no assembled scene GLB under {out}")

    if sources is None:
        from . import light_sources
        sources = light_sources.collect(out)

    # VLM tone + intensity per source (lights.json), matched by id:
    # f"{source}_{kind}_{index:02d}" — same id analyze.py assigns.
    win_by_id, pt_by_id, ambient, base_dim = {}, {}, None, 0.5
    lj = ldir / "lights.json"
    if lj.exists():
        try:
            L = json.loads(lj.read_text())
            win_by_id = {w.get("id"): w for w in L.get("windows", [])}
            pt_by_id = {p.get("id"): p for p in L.get("point_lights", [])}
            ambient = L.get("ambient")
            base_dim = float(L.get("base_dim", 0.5))
        except Exception:
            pass

    blender_lights = []
    for s in sources:
        kind = s.get("kind", "lamp")
        pos = s.get("position_m")
        if not pos:
            continue
        sid = f"{s.get('source')}_{kind}_{int(s.get('index', 0)):02d}"
        if kind == "window":
            est = win_by_id.get(sid, {})
            on = bool(est.get("daylight_on", True))
        else:
            est = pt_by_id.get(sid, {})
            on = bool(est.get("on", True))
        if not on:  # VLM judged this source unlit
            print(f"[blender] source {sid} off (VLM) — skipped")
            continue
        color = est.get("color") or _KIND_COLOR.get(kind, [1.0, 1.0, 1.0])
        if os.environ.get("SCENEWEAVE_LIGHT_WHITE"):
            color = [1.0, 1.0, 1.0]  # force neutral white illumination
        power = _power(kind, est.get("intensity"), power_scale)
        light = {
            "kind": kind,
            "position_m": list(map(float, pos)),
            "color": list(color),
            "power": power,
        }
        # A window/curtain must light its FULL opening — pass the real
        # size_m (width, height) so the Cycles area light spans the whole
        # window instead of the 0.8×2.0 default (which lit only the top).
        if kind == "window":
            sm = s.get("size_m") or {}
            w = float(sm.get("width_m") or 0.8)
            h = float(sm.get("height_m") or 2.0)
            light["size"] = [max(0.3, w), max(0.3, h)]
        blender_lights.append(light)

    # Match render_pyrender's camera selection (camera_vggt.json first) so the
    # Blender view is identical to the rest of the lighting previews.
    cam = None
    for cand in (out / "camera_vggt.json", out / "camera.json"):
        if cand.exists():
            cam = json.loads(cand.read_text())
            print(f"[blender] camera: {cand.name}")
            break
    if cam is None:
        raise FileNotFoundError(f"no camera json under {out}")

    # Ceiling-fixture geometry (lights are separate; this is just the visible mesh).
    extra = []
    csr = out / "ceiling" / "segment_results.json"
    if csr.exists():
        try:
            for seg in json.loads(csr.read_text()).get("segments", []):
                plc = seg.get("placement") or {}
                gf = seg.get("glb_file")
                if gf and "position_m" in plc:
                    # Only re-add the ceiling fixture as an extra when an emissive
                    # lit_<stem>.glb copy exists — the whole point of the swap is to
                    # replace the dark baked fixture with a glowing one. Without a
                    # lit_ copy, keep the fixture already baked into scene_full.glb
                    # (correctly placed): re-adding the raw ceiling GLB double-offsets
                    # it by position_m in scaled scene_full coords, hiding it above
                    # the ceiling (the baked copy is then also dedup'd → fixture
                    # disappears entirely).
                    lit = ldir / "objects" / f"lit_{Path(gf).stem}.glb"
                    if lit.exists():
                        extra.append({"glb": str(lit),
                                      "position_m": plc["position_m"],
                                      "scale": plc.get("scale", [1, 1, 1])})
        except Exception:
            pass

    # Carpet: a flat textured floor plane baked separately, because the carpet
    # placement carries no GLB (is_carpet → 2D floor overlay only), so the
    # path-traced scene would otherwise have no rug. See lighting_module/bake_carpet.py.
    # AUTO-BAKE: if the scene has an is_carpet placement but no carpet.glb yet,
    # bake it now so every carpet scene gets its rug without a manual step (the
    # bake also auto-handles rug-on-white masks). Disable with SCENEWEAVE_NO_CARPET.
    carpet_glb = ldir / "objects" / "carpet.glb"
    if (not carpet_glb.exists()
            and os.environ.get("SCENEWEAVE_NO_CARPET") != "1"):
        try:
            _fp = out / "furniture" / "furniture_placements.json"
            if _fp.exists() and any(
                    e.get("is_carpet") for e in json.loads(_fp.read_text())):
                from lighting_module.bake_carpet import bake as _bake_carpet
                print("[blender] no carpet.glb — auto-baking carpet …")
                _bake_carpet(str(out))
        except Exception as _e:
            print(f"[blender] carpet auto-bake skipped: {_e}")
    if carpet_glb.exists():
        extra.append({"glb": str(carpet_glb),
                      "position_m": [0.0, 0.0, 0.0], "scale": [1, 1, 1]})

    # assemble_scene bakes the ceiling fixture INTO scene_full.glb, and we also
    # re-add it above as the emissive lit_ extra — so without removing the baked
    # copy the fixture renders twice ("two ceiling lights"). Strip the baked
    # ceiling-object geom(s) into a deduped temp GLB (the shell ceiling is named
    # plain "ceiling"/"placements_ceiling" and is preserved).
    has_ceiling_extra = any(
        ("lit_" in Path(e["glb"]).name) or ("/ceiling/" in e["glb"].replace("\\", "/"))
        for e in extra)
    if has_ceiling_extra:
        try:
            import trimesh
            _sc = trimesh.load(str(scene_glb), force="scene")
            _drop = [n for n in _sc.geometry
                     if n.lower() not in ("ceiling", "placements_ceiling")
                     and (n.lower().startswith("ceiling_ceiling")
                          or any(k in n.lower() for k in
                                 ("chandelier", "pendant", "ceiling_light",
                                  "ceiling_lamp", "ceiling_fan", "sconce")))]
            if _drop:
                _sc.delete_geometry(_drop)
                _dedup = ldir / "_scene_dedup.glb"
                _sc.export(str(_dedup))
                scene_glb = _dedup
                print(f"[blender] stripped baked ceiling duplicate(s): {_drop}")
        except Exception as _e:
            print(f"[blender] ceiling-dedup skipped: {_e}")

    # assemble_scene adds each wall object TWICE — once as "wallobj_NN_*" (wall
    # stage) and once as "placements_wallobj_NN_*" (decoration assembly). The two
    # coincident copies sit flat ON the wall plane, so the path tracer z-fights
    # between them AND the wall shell → the art renders as a see-through mottle.
    # Drop the duplicate copy and nudge the kept wall objects a few cm off the
    # wall (toward room centre) so nothing is coplanar.
    try:
        import trimesh
        import numpy as _np
        _sc = trimesh.load(str(scene_glb), force="scene")
        _wall = [n for n in _sc.geometry if "wallobj" in n.lower()]
        _plain = {n[len("placements_"):]: n for n in _wall
                  if n.lower().startswith("placements_")}
        _dupes = [_plain[b] for b in _plain
                  if any(w.lower() == b.lower() for w in _wall)]
        if _dupes:
            _sc.delete_geometry(_dupes)
            _wall = [n for n in _sc.geometry if "wallobj" in n.lower()]
        _ctr = _np.array(_room_center(scene_glb), float)
        _moved = 0
        for _node in list(_sc.graph.nodes_geometry):
            _T, _g = _sc.graph[_node]
            if "wallobj" not in _g.lower():
                continue
            _T = _np.array(_T)
            _gc = _sc.geometry[_g].bounds.mean(axis=0) + _T[:3, 3]
            _dir = _ctr - _gc
            _dir[1] = 0.0  # keep height; slide horizontally toward interior
            _nrm = _np.linalg.norm(_dir)
            if _nrm > 1e-3:
                _T[:3, 3] += (_dir / _nrm) * 0.04
                _sc.graph.update(frame_to=_node, matrix=_T)
                _moved += 1
        if _dupes or _moved:
            _wd = ldir / "_scene_walldedup.glb"
            _sc.export(str(_wd))
            scene_glb = _wd
            print(f"[blender] wall-art dedup: dropped {len(_dupes)} dup copy(ies), "
                  f"offset {_moved} wall object(s) off the wall")
    except Exception as _e:
        print(f"[blender] wall-art dedup skipped: {_e}")

    cfg = {
        "scene_glb": str(scene_glb),
        "extra_objects": extra,
        "camera": {
            "position_m": cam["position_m"],
            "look_at_m": cam["look_at_m"],
            "up": cam.get("up", [0, 1, 0]),
            "hfov_deg": cam.get("hfov_deg", 60.0),
        },
        "room_center_m": _room_center(scene_glb),
        "glass_objects": _collect_glass_objects(out),
        "lights": blender_lights,
        "world": _world_from_ambient(ambient, base_dim),
        "light_white": bool(os.environ.get("SCENEWEAVE_LIGHT_WHITE")),
        "resolution": [int(cam.get("width_px", 625)),
                       int(cam.get("height_px", 350))],
        "samples": int(samples),
        "view_transform": "AgX",
        "exposure": float(exposure),
        # dims self-lit surfaces (e.g. blown-out emissive window glass) without
        # touching diffuse-lit geometry; per-scene via SCENEWEAVE_EMISSIVE_SCALE.
        "emissive_scale": float(os.environ.get("SCENEWEAVE_EMISSIVE_SCALE", 1.0)),
        "output_png": str(output_path or (ldir / "render_blender.png")),
    }
    cfg_path = ldir / "_blender_cfg.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    blender = _find_blender()
    script = Path(__file__).parent / "_blender_cycles.py"
    cmd = [blender, "--background", "--python", str(script), "--", str(cfg_path)]
    print(f"[blender] {len(blender_lights)} light(s) → {cfg['output_png']}")
    subprocess.run(cmd, check=True)

    # ── exposure match against the reference photograph ──────────────────────
    # Nothing in the pipeline ever compared the FINAL render to the photo: the
    # refine loop grades a 2D lights-only composite, so the Cycles output's
    # brightness was never checked against anything.  Measured over six scenes,
    # a scene with lights renders 1.2-1.8x brighter than its reference, and a
    # scene that lost its light sources renders at 0.05-0.10x — near-black.
    # Luminance is a deterministic thing to measure, so measure it and correct
    # in EV rather than asking a VLM to eyeball it again.
    if os.environ.get("SCENEWEAVE_NO_EXPOSURE_MATCH") == "1":
        return cfg["output_png"]
    try:
        import numpy as _np
        from PIL import Image as _Im

        def _lum(p):
            im = _Im.open(p).convert("RGB")
            im.thumbnail((900, 900))
            return float((_np.asarray(im, float) / 255.0
                          @ [0.2126, 0.7152, 0.0722]).mean())

        # The scene dir keeps a canonical copy of the input named after itself
        # (main.py copies it there so --rerun-dir is self-contained).
        _ref = None
        for _ext in (".png", ".jpg", ".jpeg", ".webp", ".avif"):
            _c = out / f"{out.name}{_ext}"
            if _c.exists():
                _ref = _c
                break
        if _ref is None:
            _cands = [p for p in out.glob("*") if p.suffix.lower() in
                      (".png", ".jpg", ".jpeg") and not p.name.startswith(("render_", "manhattan_"))
                      and "texture" not in p.name and "empty_room" not in p.name]
            _ref = _cands[0] if _cands else None
        if _ref is None:
            print("[blender] exposure match skipped: no reference image found")
        if _ref and Path(_ref).exists():
            l_ref, l_out = _lum(_ref), _lum(cfg["output_png"])
            ratio = l_out / max(l_ref, 1e-6)
            _TOL = float(os.environ.get("SCENEWEAVE_EXPOSURE_TOL", "0.25"))
            if abs(ratio - 1.0) > _TOL and l_out > 1e-5:
                ev = float(_np.clip(_np.log2(max(l_ref, 1e-4) / max(l_out, 1e-4)),
                                    -3.0, 3.0))
                print(f"[blender] exposure match: render/ref luminance = "
                      f"{ratio:.2f} → re-rendering at {ev:+.2f} EV")
                cfg["exposure"] = float(exposure) + ev
                cfg_path.write_text(json.dumps(cfg, indent=2))
                subprocess.run(cmd, check=True)
                print(f"[blender] after match: ratio = "
                      f"{_lum(cfg['output_png']) / max(l_ref, 1e-6):.2f}")
            else:
                print(f"[blender] exposure ok: render/ref luminance = {ratio:.2f}")
    except Exception as _ee:
        print(f"[blender] exposure match skipped: {_ee}")
    return cfg["output_png"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cycles physical lighting render.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--power-scale", type=float, default=1.0)
    a = ap.parse_args()
    render(a.output_dir, samples=a.samples, exposure=a.exposure,
           power_scale=a.power_scale)


if __name__ == "__main__":
    main()
