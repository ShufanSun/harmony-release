"""
material_properties.py — VLM material estimation for CEILING objects.

Mirrors the furniture/decoration material stages: classifies the ceiling
fixture's material (track/pendant light, chandelier, fan, …) and writes the PBR
factors onto its GLB (metallic/roughness + KHR_materials_ior, plus
transmission/volume for glass/acrylic shades) so the ceiling composite shades it
correctly (glossy metal track light, glass diffuser).  Runs BEFORE
ceiling_placement so the composited fixture already carries its material.

Reuses the furniture material core (estimate_material / apply_to_glb).

Usage:
    python -m object_placement.ceiling.material_properties \\
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


def run(output_dir: str | Path, verbose: bool = True) -> dict:
    """Estimate + apply materials for every generated ceiling object GLB.

    Iterates <output_dir>/ceiling/objects/inpaint_*.glb, uses each object's
    inpainted image for the VLM call, writes the factors into the GLB + a
    ceiling/material_properties.json sidecar.  No scene re-bake — ceiling_placement
    re-composites the materialled fixture afterwards."""
    out = Path(output_dir)
    cdir = out / "ceiling"
    obj_dir = cdir / "objects"
    inpaint_dir = cdir / "inpainted"
    if not obj_dir.exists():
        raise FileNotFoundError(f"ceiling/objects not found: {obj_dir}")

    glbs = sorted(obj_dir.glob("inpaint_*.glb"))
    materials: dict[str, dict] = {}
    print(f"[ceil-material] estimating materials for {len(glbs)} ceiling object(s) …")
    for glb in glbs:
        m = re.match(r"inpaint_(\d+)_(.+)", glb.stem)
        idx = int(m.group(1)) if m else -1
        otype = (m.group(2) if m else glb.stem).replace("_", " ")
        img = inpaint_dir / f"{glb.stem}.png"
        if not img.exists():
            cand = list(inpaint_dir.glob(f"{glb.stem}.*"))
            img = cand[0] if cand else None
        if img is None:
            print(f"  [{idx:02d} {otype:<14}] no inpaint image — skipping")
            continue
        props = estimate_material(img, otype)
        applied = apply_to_glb(glb, props)
        props["glb"] = glb.name
        props["applied"] = applied
        materials[glb.name] = props
        if verbose:
            print(f"  [{idx:02d} {otype:<14}] {props['material_category']:<8} "
                  f"rough={props['roughness']:.2f} metal={props['metallic']:.2f} "
                  f"ior={props['ior']:.2f} transm={props['transmission']:.2f}"
                  f"  → {'GLB updated' if applied else 'GLB unchanged'}")

    sidecar = cdir / "material_properties.json"
    sidecar.write_text(json.dumps(materials, indent=2))
    print(f"[ceil-material] wrote {sidecar.relative_to(out)} ({len(materials)} objects)")
    return materials


def main():
    ap = argparse.ArgumentParser(description="VLM material estimation for ceiling objects.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
