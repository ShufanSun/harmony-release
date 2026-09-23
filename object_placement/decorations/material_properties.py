"""
material_properties.py — VLM material estimation for DECORATION objects.

POST-placement finishing pass (mirrors the furniture material stage): for each
generated decoration GLB (throw pillow, blanket, books, bowl, vase, potted
plant, lantern, …) the VLM classifies the dominant material and estimates the
PBR + physical property set, which is written onto the GLB material
(metallic/roughness + KHR_materials_ior, plus transmission/volume for glass
lanterns) and a sidecar.  Then scene_with_decoration.glb is re-baked from the
saved placement JSON so the assembled scene shades the decorations correctly
(matte fabric pillows, paper books, ceramic bowl, glass lantern).

Reuses the furniture material core (estimate_material / apply_to_glb).

Usage:
    python -m object_placement.decorations.material_properties \\
        --output-dir outputs/_HARMONY300/complicated/living_room9
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from object_placement.furniture.material_properties import (
    apply_to_glb,
    estimate_material,
)


def run(output_dir: str | Path, verbose: bool = True,
        rebake_scene: bool = False) -> dict:
    """Estimate + apply materials for every generated decoration object, then
    (with rebake_scene=True) re-bake scene_with_decoration.glb from the saved
    decoration placement JSON — no re-placement, just re-reads the materialled
    decoration GLBs + their transforms."""
    out = Path(output_dir)
    ddir = out / "decorations"
    obj_dir = ddir / "objects"
    inpaint_dir = ddir / "inpainted"
    if not obj_dir.exists():
        raise FileNotFoundError(f"decorations/objects not found: {obj_dir}")

    glbs = sorted(p for p in obj_dir.glob("decor_*.glb"))
    materials: dict[str, dict] = {}
    print(f"[decor-material] estimating materials for {len(glbs)} decoration object(s) …")
    for glb in glbs:
        m = re.match(r"decor_(\d+)_(.+)", glb.stem)
        idx = int(m.group(1)) if m else -1
        phrase = (m.group(2) if m else glb.stem).replace("_", " ")
        phrase = phrase.strip().strip('"').strip("!").strip() or "object"
        cand = sorted(inpaint_dir.glob(f"inpaint_{idx:02d}_*.png"))
        img = cand[0] if cand else None
        if img is None:
            print(f"  [{idx:02d} {phrase:<14}] no inpaint image — skipping")
            continue
        props = estimate_material(img, phrase)
        applied = apply_to_glb(glb, props)
        props["glb"] = glb.name
        props["applied"] = applied
        materials[glb.name] = props
        if verbose:
            print(f"  [{idx:02d} {phrase:<14}] {props['material_category']:<8} "
                  f"rough={props['roughness']:.2f} metal={props['metallic']:.2f} "
                  f"ior={props['ior']:.2f} transm={props['transmission']:.2f}"
                  f"  → {'GLB updated' if applied else 'GLB unchanged'}")

    sidecar = ddir / "material_properties.json"
    sidecar.write_text(json.dumps(materials, indent=2))
    print(f"[decor-material] wrote {sidecar.relative_to(out)} ({len(materials)} objects)")

    if rebake_scene:
        plc = ddir / "placements" / "decoration_placements.json"
        if plc.exists():
            placements = json.loads(plc.read_text())
            try:
                from object_placement.decorations.place_decorations import (
                    _export_scene_with_decorations as _export,
                )
                _export(out, placements, ddir / "placements")
                print("[decor-material] re-baked scene_with_decoration.glb "
                      "(materials + textured walls)")
            except Exception as e:
                print(f"[decor-material] scene re-bake skipped: {e}")
            # Re-render the 2D composite from the now-materialled decoration GLBs
            # so render_decorations_placed.png (which the ceiling phase composites
            # over) reflects the materials/textures — not the pre-material
            # placement render.
            try:
                from object_placement.decorations.place_decorations import (
                    _render_final as _render_final_decor,
                )
                _render_final_decor(out, placements, ddir / "placements",
                                    inpaint_dir=ddir / "inpainted")
                print("[decor-material] re-rendered render_decorations_placed.png "
                      "(materialled decorations)")
            except Exception as e:
                print(f"[decor-material] re-render skipped: {e}")
        else:
            print("[decor-material] no decoration_placements.json — scene not re-baked")
    return materials


def main():
    ap = argparse.ArgumentParser(description="VLM material estimation for decoration objects.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
