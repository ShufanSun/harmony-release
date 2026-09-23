"""
material_properties.py — VLM-driven PBR material estimation for wall-mounted objects.

For each segmented + generated wall object (window, curtain, art, light, …) this
asks the VLM to classify the object's material and estimate physically-based
render (PBR) properties — material category, roughness, metallic, and IOR — then
writes them onto the object's GLB material (pbrMetallicRoughness.metallicFactor /
roughnessFactor + the KHR_materials_ior extension) and a sidecar
``material_properties.json``.

The base-colour texture (from Hunyuan3D or the window-plane path) is left
untouched — only the scalar PBR factors are set, so the assembled scene gets
correct shading (matte fabric curtains, glossy metal sconces, glass windows).

Usage:
    python -m object_placement.wall_mounted.material_properties \\
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


# ── Per-category PBR defaults (used to sanity-clamp / fill VLM gaps) ───────────
# metallic, roughness, ior.  These are reasonable physical baselines; the VLM
# refines roughness/metallic within them.
_CATEGORY_DEFAULTS: dict[str, dict] = {
    "metal":    {"metallic": 1.0, "roughness": 0.35, "ior": 2.5},
    "glass":    {"metallic": 0.0, "roughness": 0.03, "ior": 1.5},
    "fabric":   {"metallic": 0.0, "roughness": 0.9,  "ior": 1.4},
    "wood":     {"metallic": 0.0, "roughness": 0.6,  "ior": 1.45},
    "painted":  {"metallic": 0.0, "roughness": 0.7,  "ior": 1.45},
    "ceramic":  {"metallic": 0.0, "roughness": 0.25, "ior": 1.5},
    "plastic":  {"metallic": 0.0, "roughness": 0.4,  "ior": 1.46},
    "paper":    {"metallic": 0.0, "roughness": 0.85, "ior": 1.45},
    "leather":  {"metallic": 0.0, "roughness": 0.7,  "ior": 1.45},
    "stone":    {"metallic": 0.0, "roughness": 0.6,  "ior": 1.5},
    "other":    {"metallic": 0.0, "roughness": 0.6,  "ior": 1.45},
}
_CATEGORIES = sorted(_CATEGORY_DEFAULTS)


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
                      timeout: int = 90) -> dict:
    """Ask the VLM for the object's PBR material. Returns a dict with keys
    material_category, roughness, metallic, ior (always populated — falls back to
    per-category / per-type defaults on any failure)."""
    b64 = _encode_png_b64(Path(image_path))
    prompt = (
        f"The image shows a single {obj_type} isolated on a plain grey background.\n"
        "You are a physically-based-rendering (PBR) material expert. Estimate the "
        "object's surface material. Respond with ONLY a JSON object, no prose:\n"
        "{\n"
        f'  "material_category": "<one of: {", ".join(_CATEGORIES)}>",\n'
        '  "roughness": <float 0.0-1.0>,   // 0 = mirror-smooth/glossy, 1 = fully matte\n'
        '  "metallic":  <float 0.0-1.0>,   // 1 = bare metal, 0 = dielectric/non-metal\n'
        '  "ior":       <float 1.0-2.5>    // index of refraction\n'
        "}\n"
        "Guidance:\n"
        "- bare/brushed/polished metal (brass, steel, chrome, iron): metallic≈1.0, "
        "roughness 0.1-0.5, ior 2.0-2.5\n"
        "- glass / glossy glaze: metallic 0, roughness 0.0-0.1, ior ~1.5\n"
        "- fabric / curtain / cloth / linen: metallic 0, roughness 0.8-1.0, ior ~1.4\n"
        "- painted wood, canvas, paper, print: metallic 0, roughness 0.6-0.9, ior ~1.45\n"
        "- ceramic / glazed: metallic 0, roughness 0.2-0.4, ior ~1.5\n"
        "Judge from the visible sheen/reflections in the image. For a window assembly "
        "report the dominant FRAME material (the glass is handled separately)."
    )
    payload = {
        "model": "qwen3",
        "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    cat, rough, metal, ior = None, None, None, None
    try:
        resp = _vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group()) if m else {}
        cat = str(data.get("material_category", "")).strip().lower()
        rough, metal, ior = data.get("roughness"), data.get("metallic"), data.get("ior")
    except Exception as e:
        print(f"  [material] VLM failed for {obj_type}: {e} — using defaults")

    if cat not in _CATEGORY_DEFAULTS:
        # fall back by object type when the category is missing/unknown
        cat = {"window": "glass", "door": "wood", "curtain": "fabric",
               "art": "painted", "painting": "painted", "frame": "wood",
               "light": "metal", "mirror": "glass", "shelf": "wood",
               "tv": "plastic", "clock": "plastic"}.get(obj_type, "other")
    dflt = _CATEGORY_DEFAULTS[cat]
    return {
        "material_category": cat,
        "roughness": round(_clamp(rough, 0.0, 1.0, dflt["roughness"]), 3),
        "metallic":  round(_clamp(metal, 0.0, 1.0, dflt["metallic"]),  3),
        "ior":       round(_clamp(ior,   1.0, 2.5, dflt["ior"]),       3),
    }


def apply_to_glb(glb_path: str | Path, props: dict) -> bool:
    """Write metallic/roughness factors + KHR_materials_ior onto every material of
    the GLB, preserving its base-colour texture. Returns True on success."""
    from pygltflib import GLTF2
    from pygltflib import PbrMetallicRoughness  # type: ignore
    glb_path = Path(glb_path)
    try:
        g = GLTF2().load(str(glb_path))
    except Exception as e:
        print(f"  [material] could not load {glb_path.name}: {e}")
        return False
    if not g.materials:
        # geometry has no material slot (rare) — nothing to set
        return False
    for mat in g.materials:
        if mat.pbrMetallicRoughness is None:
            mat.pbrMetallicRoughness = PbrMetallicRoughness()
        mat.pbrMetallicRoughness.metallicFactor = float(props["metallic"])
        mat.pbrMetallicRoughness.roughnessFactor = float(props["roughness"])
        ext = mat.extensions if isinstance(mat.extensions, dict) else {}
        ext["KHR_materials_ior"] = {"ior": float(props["ior"])}
        mat.extensions = ext
    used = set(g.extensionsUsed or [])
    used.add("KHR_materials_ior")
    g.extensionsUsed = sorted(used)
    g.save(str(glb_path))
    return True


def run(output_dir: str | Path, verbose: bool = True) -> dict:
    """Estimate + apply PBR materials for every generated wall-mounted object.

    Reads <output_dir>/wall_mounted/segment_results.json, uses each object's
    inpainted image for the VLM material call, writes the factors into its GLB and
    a material_properties.json sidecar. Returns the {filename: props} map."""
    out = Path(output_dir)
    wm = out / "wall_mounted"
    results_p = wm / "segment_results.json"
    if not results_p.exists():
        raise FileNotFoundError(f"segment_results.json not found: {results_p}")
    segs = json.loads(results_p.read_text()).get("segments", [])

    inpaint_dir = wm / "inpainted"
    obj_dir = wm / "objects"
    materials: dict[str, dict] = {}
    print(f"[material] estimating PBR materials for {len(segs)} wall object(s) …")
    for s in segs:
        idx, otype = s.get("index"), s.get("type", "other")
        inpaint_file = s.get("inpaint_file")
        if not inpaint_file:
            continue
        img = inpaint_dir / inpaint_file
        glb = obj_dir / (Path(inpaint_file).stem + ".glb")
        if not img.exists():
            continue
        props = estimate_material(img, otype)
        applied = apply_to_glb(glb, props) if glb.exists() else False
        props["glb"] = glb.name
        props["applied"] = applied
        materials[inpaint_file] = props
        if verbose:
            print(f"  [{idx:02d} {otype:<8}] {props['material_category']:<8} "
                  f"rough={props['roughness']:.2f} metal={props['metallic']:.2f} "
                  f"ior={props['ior']:.2f}  → {'GLB updated' if applied else 'GLB missing'}")

    sidecar = wm / "material_properties.json"
    sidecar.write_text(json.dumps(materials, indent=2))
    print(f"[material] wrote {sidecar.relative_to(out)} ({len(materials)} objects)")
    return materials


def main():
    ap = argparse.ArgumentParser(description="VLM PBR material estimation for wall objects.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
