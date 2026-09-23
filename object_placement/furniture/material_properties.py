"""
material_properties.py — VLM-driven material estimation for FURNITURE objects.

For each generated furniture object (sofa, chair, coffee table, side table,
plant pot, stool, …) this asks the VLM to classify the dominant surface material
and estimate a full property set:

  Render (PBR) — written onto the GLB material:
    • material_category, roughness, metallic, ior
    • transmission  → KHR_materials_transmission (glass tables, acrylic)
    • thickness_m   → KHR_materials_volume thicknessFactor (only for transmissive
                      objects, so refraction reads correctly)

  Physical — written to a sidecar ``material_properties.json`` (for physics /
  simulation / downstream reasoning, NOT glTF shading):
    • weight_kg, elasticity (0 rigid … 1 springy), thickness_m

The base-colour texture (Hunyuan bake) is left untouched — only scalar PBR
factors + extensions are set, so a fabric sofa reads matte, a glass table reads
transmissive, a metal lamp base reads glossy.

Usage:
    python -m object_placement.furniture.material_properties \\
        --output-dir outputs/_HARMONY300/complicated/living_room9
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

from object_placement.vlm_backend import vlm_post as _vlm_post


# ── Per-category property defaults (sanity-clamp + fill VLM gaps) ──────────────
# metallic, roughness, ior, transmission, elasticity, density (kg/m^3 — combined
# with the object's real-world volume to estimate weight when the VLM can't).
_CATEGORY_DEFAULTS: dict[str, dict] = {
    "fabric":   {"metallic": 0.0, "roughness": 0.92, "ior": 1.40, "transmission": 0.0, "elasticity": 0.65, "density": 120},
    "leather":  {"metallic": 0.0, "roughness": 0.55, "ior": 1.45, "transmission": 0.0, "elasticity": 0.45, "density": 180},
    "wood":     {"metallic": 0.0, "roughness": 0.60, "ior": 1.45, "transmission": 0.0, "elasticity": 0.05, "density": 600},
    "metal":    {"metallic": 1.0, "roughness": 0.35, "ior": 2.50, "transmission": 0.0, "elasticity": 0.10, "density": 7500},
    "glass":    {"metallic": 0.0, "roughness": 0.05, "ior": 1.50, "transmission": 0.90, "elasticity": 0.02, "density": 2500},
    "plastic":  {"metallic": 0.0, "roughness": 0.40, "ior": 1.46, "transmission": 0.0, "elasticity": 0.25, "density": 950},
    "ceramic":  {"metallic": 0.0, "roughness": 0.25, "ior": 1.50, "transmission": 0.0, "elasticity": 0.03, "density": 2300},
    "stone":    {"metallic": 0.0, "roughness": 0.45, "ior": 1.50, "transmission": 0.0, "elasticity": 0.02, "density": 2600},
    "rattan":   {"metallic": 0.0, "roughness": 0.80, "ior": 1.45, "transmission": 0.0, "elasticity": 0.30, "density": 350},
    "foam":     {"metallic": 0.0, "roughness": 0.95, "ior": 1.40, "transmission": 0.0, "elasticity": 0.85, "density": 60},
    "paper":    {"metallic": 0.0, "roughness": 0.85, "ior": 1.45, "transmission": 0.0, "elasticity": 0.10, "density": 700},
    "other":    {"metallic": 0.0, "roughness": 0.60, "ior": 1.45, "transmission": 0.0, "elasticity": 0.20, "density": 700},
}
_CATEGORIES = sorted(_CATEGORY_DEFAULTS)

# Fall-back category by furniture type when the VLM category is missing/unknown.
_TYPE_CATEGORY = {
    "sofa": "fabric", "couch": "fabric", "sectional": "fabric", "loveseat": "fabric",
    "armchair": "fabric", "chair": "fabric", "accent_chair": "fabric", "ottoman": "fabric",
    "stool": "wood", "bench": "wood", "coffee_table": "wood", "side_table": "wood",
    "end_table": "wood", "console": "wood", "desk": "wood", "cabinet": "wood",
    "bookcase": "wood", "shelf": "wood", "dresser": "wood", "nightstand": "wood",
    "bed": "fabric", "tv_stand": "wood", "plant": "ceramic", "leafy_plant": "ceramic",
    "tree": "ceramic", "lamp": "metal", "floor_lamp": "metal",
}


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _encode_png_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def estimate_material(image_path: str | Path, obj_type: str,
                      size_m: dict | None = None, timeout: int = 90) -> dict:
    """Ask the VLM for the furniture object's full material property set.

    Always returns a fully-populated dict (material_category, roughness,
    metallic, ior, transmission, weight_kg, thickness_m, elasticity), falling
    back to per-category / per-type physical baselines on any failure."""
    b64 = _encode_png_b64(Path(image_path))
    size_hint = ""
    if size_m:
        size_hint = (f" Its real-world bounding size is about "
                     f"{size_m.get('width_m', 0):.2f}×{size_m.get('height_m', 0):.2f}×"
                     f"{size_m.get('depth_m', 0):.2f} m — use it to estimate weight.")
    prompt = (
        f"The image shows a single piece of furniture ({obj_type}) isolated on a "
        f"plain grey background.{size_hint}\n"
        "You are a physically-based-rendering (PBR) + physical-properties expert. "
        "Estimate the object's DOMINANT surface material and physical properties. "
        "Respond with ONLY a JSON object, no prose:\n"
        "{\n"
        f'  "material_category": "<one of: {", ".join(_CATEGORIES)}>",\n'
        '  "roughness":    <float 0.0-1.0>,   // 0 = glossy/mirror, 1 = fully matte\n'
        '  "metallic":     <float 0.0-1.0>,   // 1 = bare metal, 0 = non-metal\n'
        '  "ior":          <float 1.0-2.5>,   // index of refraction\n'
        '  "transmission": <float 0.0-1.0>,   // 1 = clear glass/acrylic, 0 = opaque\n'
        '  "weight_kg":    <float>,           // whole-object mass estimate\n'
        '  "thickness_m":  <float>,           // dominant panel / shell thickness\n'
        '  "elasticity":   <float 0.0-1.0>    // 0 = rigid (wood/metal/stone), 1 = springy (foam cushion)\n'
        "}\n"
        "Guidance:\n"
        "- upholstered sofa/armchair/cushion: fabric (or leather), roughness 0.8-1.0, "
        "metallic 0, transmission 0, elasticity 0.6-0.9 (soft cushions)\n"
        "- solid wood table/cabinet/stool: wood, roughness 0.5-0.8, metallic 0, "
        "elasticity 0.02-0.1\n"
        "- glass/acrylic table top: glass, roughness 0.0-0.1, transmission 0.8-0.95, "
        "ior ~1.5, elasticity ~0\n"
        "- metal frame/legs/lamp: metal, metallic ~1.0, roughness 0.2-0.5, elasticity ~0.1\n"
        "- ceramic/stone planter or marble top: ceramic/stone, roughness 0.2-0.5, elasticity ~0.03\n"
        "Judge material from the visible sheen/reflections; if the piece mixes "
        "materials report the one covering the MOST surface area."
    )
    payload = {
        "model": "qwen3",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    data: dict = {}
    try:
        resp = _vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group()) if m else {}
    except Exception as e:
        print(f"  [material] VLM failed for {obj_type}: {e} — using defaults")

    cat = str(data.get("material_category", "")).strip().lower()
    if cat not in _CATEGORY_DEFAULTS:
        cat = _TYPE_CATEGORY.get(obj_type, "other")
    d = _CATEGORY_DEFAULTS[cat]

    # Weight: trust the VLM if plausible, else density × real volume (m^3).
    weight = data.get("weight_kg")
    if weight is None and size_m:
        vol = max(1e-3, float(size_m.get("width_m", 0)) * float(size_m.get("height_m", 0))
                  * float(size_m.get("depth_m", 0)))
        # furniture is mostly hollow/framed, so apply a 0.25 fill factor.
        weight = d["density"] * vol * 0.25
    return {
        "material_category": cat,
        "roughness":    round(_clamp(data.get("roughness"),    0.0, 1.0, d["roughness"]),    3),
        "metallic":     round(_clamp(data.get("metallic"),     0.0, 1.0, d["metallic"]),     3),
        "ior":          round(_clamp(data.get("ior"),          1.0, 2.5, d["ior"]),          3),
        "transmission": round(_clamp(data.get("transmission"), 0.0, 1.0, d["transmission"]), 3),
        "weight_kg":    round(_clamp(weight,                   0.1, 500.0, 10.0),            2),
        "thickness_m":  round(_clamp(data.get("thickness_m"),  0.002, 0.5, 0.03),            4),
        "elasticity":   round(_clamp(data.get("elasticity"),   0.0, 1.0, d["elasticity"]),   3),
    }


def apply_to_glb(glb_path: str | Path, props: dict) -> bool:
    """Write metallic/roughness + KHR_materials_ior, and (for transmissive
    objects) KHR_materials_transmission + KHR_materials_volume, onto every
    material of the GLB.  Base-colour texture is preserved.  Returns True on
    success."""
    from pygltflib import GLTF2, PbrMetallicRoughness  # type: ignore
    glb_path = Path(glb_path)
    try:
        g = GLTF2().load(str(glb_path))
    except Exception as e:
        print(f"  [material] could not load {glb_path.name}: {e}")
        return False
    if not g.materials:
        return False
    transmissive = float(props.get("transmission", 0.0)) > 0.01
    for mat in g.materials:
        if mat.pbrMetallicRoughness is None:
            mat.pbrMetallicRoughness = PbrMetallicRoughness()
        mat.pbrMetallicRoughness.metallicFactor = float(props["metallic"])
        mat.pbrMetallicRoughness.roughnessFactor = float(props["roughness"])
        ext = mat.extensions if isinstance(mat.extensions, dict) else {}
        ext["KHR_materials_ior"] = {"ior": float(props["ior"])}
        if transmissive:
            ext["KHR_materials_transmission"] = {"transmissionFactor": float(props["transmission"])}
            # Volume thickness lets the transmission refract; use the panel thickness.
            ext["KHR_materials_volume"] = {"thicknessFactor": float(props["thickness_m"])}
        mat.extensions = ext
    used = set(g.extensionsUsed or [])
    used.add("KHR_materials_ior")
    if transmissive:
        used.update({"KHR_materials_transmission", "KHR_materials_volume"})
    g.extensionsUsed = sorted(used)
    g.save(str(glb_path))
    return True


def run(output_dir: str | Path, verbose: bool = True,
        rebake_scene: bool = False) -> dict:
    """Estimate + apply materials for every generated furniture object.

    Iterates <output_dir>/furniture/objects/*.glb, uses each object's inpainted
    image for the VLM call, writes render factors into the GLB and ALL props
    (incl. physical) into furniture/material_properties.json.  Returns the
    {filename: props} map.

    POST-placement step: with ``rebake_scene=True`` it re-bakes
    scene_with_furniture.glb from the saved placement JSON after tagging the
    object GLBs, so the assembled scene picks up the materials WITHOUT a full
    placement re-compute (placement-first is easier to debug)."""
    out = Path(output_dir)
    fdir = out / "furniture"
    obj_dir = fdir / "objects"
    inpaint_dir = fdir / "inpainted"
    if not obj_dir.exists():
        raise FileNotFoundError(f"furniture/objects not found: {obj_dir}")

    # Size hints (real-world WxHxD) per object, keyed by GLB stem, if available.
    sizes: dict[str, dict] = {}
    plc = fdir / "furniture_placements.json"
    if plc.exists():
        try:
            for o in json.loads(plc.read_text()):
                gp = o.get("glb_path") or ""
                stem = Path(gp).stem if gp else None
                if stem and isinstance(o.get("size_m"), dict):
                    sizes[stem] = o["size_m"]
        except Exception:
            pass

    glbs = sorted(p for p in obj_dir.glob("*.glb") if "carpet" not in p.name.lower())
    materials: dict[str, dict] = {}
    print(f"[material] estimating materials for {len(glbs)} furniture object(s) …")
    for glb in glbs:
        stem = glb.stem                      # e.g. inpaint_00_sofa
        m = re.match(r"inpaint_(\d+)_(.+)", stem)
        idx = int(m.group(1)) if m else -1
        otype = (m.group(2) if m else stem).replace("-", "_")
        img = inpaint_dir / f"{stem}.png"
        if not img.exists():
            # fall back to any image with the same stem
            cand = list(inpaint_dir.glob(f"{stem}.*"))
            img = cand[0] if cand else None
        if img is None or not img.exists():
            print(f"  [{idx:02d} {otype:<13}] no inpaint image — skipping")
            continue
        props = estimate_material(img, otype, size_m=sizes.get(stem))
        applied = apply_to_glb(glb, props)
        props["glb"] = glb.name
        props["applied"] = applied
        materials[glb.name] = props
        if verbose:
            print(f"  [{idx:02d} {otype:<13}] {props['material_category']:<8} "
                  f"rough={props['roughness']:.2f} metal={props['metallic']:.2f} "
                  f"ior={props['ior']:.2f} transm={props['transmission']:.2f} "
                  f"wt={props['weight_kg']:.1f}kg elas={props['elasticity']:.2f}"
                  f"  → {'GLB updated' if applied else 'GLB unchanged'}")

    # ── Harmonise matching seating sets ──────────────────────────────────────
    # Sofas/couches in a room are usually a matching SET of the same upholstery,
    # and fabric is easily misread as leather from a single sheen cue.  If any
    # same-type seating piece reads fabric, force the whole set to fabric (keep a
    # high matte roughness + soft elasticity) so a matching pair doesn't end up
    # split fabric/leather.
    _SEATING = {"sofa", "couch", "sectional", "loveseat"}

    def _type_of(glbname: str) -> str:
        mm = re.match(r"inpaint_\d+_(.+)\.glb", glbname)
        return (mm.group(1) if mm else "").replace("-", "_")

    fab = _CATEGORY_DEFAULTS["fabric"]
    for _t in _SEATING:
        grp = [(k, v) for k, v in materials.items() if _type_of(k) == _t]
        cats = {v["material_category"] for _, v in grp}
        if len(grp) >= 2 and "fabric" in cats and (cats - {"fabric"}):
            for k, v in grp:
                if v["material_category"] == "fabric":
                    continue
                v["material_category"] = "fabric"
                v["roughness"] = round(max(v["roughness"], 0.85), 3)
                v["metallic"] = 0.0
                v["ior"] = fab["ior"]
                v["transmission"] = 0.0
                v["elasticity"] = round(max(v["elasticity"], 0.6), 3)
                v["applied"] = apply_to_glb(obj_dir / v["glb"], v)
                print(f"  [harmonize] {v['glb']}: {_t} → fabric (match seating set)")

    sidecar = fdir / "material_properties.json"
    sidecar.write_text(json.dumps(materials, indent=2))
    print(f"[material] wrote {sidecar.relative_to(out)} ({len(materials)} objects)")

    # Re-bake scene_with_furniture.glb from the saved placement JSON so the
    # assembled scene picks up the new materials — no placement re-compute, just
    # re-reads the now-materialled object GLBs + their stored transforms (and
    # re-textures the walls via geo_save's retexture step).
    if rebake_scene:
        plc = fdir / "furniture_placements.json"
        if plc.exists():
            try:
                from object_placement.furniture.place_furniture_vggt import (
                    save_placed_geometry as _save_geo,
                )
                _save_geo(out, json.loads(plc.read_text()))
                print("[material] re-baked scene_with_furniture.glb "
                      "(materials + textured walls)")
            except Exception as e:
                print(f"[material] scene re-bake skipped: {e}")
        else:
            print("[material] no furniture_placements.json — scene not re-baked")
    return materials


def main():
    ap = argparse.ArgumentParser(description="VLM material estimation for furniture objects.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
