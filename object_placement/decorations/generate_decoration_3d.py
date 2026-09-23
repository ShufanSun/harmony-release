"""
generate_decoration_3d.py — generate 3D GLB models for decoration objects via
Hunyuan3D (the only 3D-generation backend in this release — Hunyuan3D is
required, not optional).

Reads segment_results.json from <output_dir>/decorations/, uses each
segment's inpaint_file (from decorations/inpainted/) to generate a GLB.
Outputs saved to decorations/objects/.

On a Hunyuan3D 404 (the known can't-mesh-this signature — glass/transparent
objects deadlock marching cubes), this re-inpaints the segment with an
opaque-material qualifier appended to its phrase and retries once.

Usage:
    python -m object_placement.decorations.generate_decoration_3d \
        --output-dir outputs/office8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Re-use generation helpers from the furniture pipeline
from object_placement.furniture.object_generation import (
    HunyuanUnmeshableError,
    _generate_hunyuan,
    _hunyuan_available,
)


def _reinpaint_opaque(output_dir: Path, seg_idx: int) -> Path | None:
    """Re-inpaint one decoration segment with an opaque-material qualifier
    and return its new image path, or None on failure."""
    from object_placement.decorations.inpaint_decorations import run as run_inpaint

    decor_dir = output_dir / "decorations"
    results_path = decor_dir / "segment_results.json"
    with open(results_path) as f:
        data = json.load(f)
    for s in data["segments"]:
        if s["seg_index"] == seg_idx:
            base = s.get("phrase", "decoration")
            s["phrase"] = (
                f"{base}, made of solid opaque material (NOT glass, NOT "
                "transparent, NOT translucent)"
            )
            break
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [reinpaint] regenerating idx={seg_idx} as opaque material …")
    run_inpaint(output_dir, indices=[seg_idx])

    with open(results_path) as f:
        data = json.load(f)
    for s in data["segments"]:
        if s["seg_index"] == seg_idx and s.get("inpaint_file"):
            return decor_dir / "inpainted" / s["inpaint_file"]
    return None


def run(
    output_dir: str | Path,
    indices: list[int] | None = None,
    texture: bool = True,
) -> Path:
    out_dir   = Path(output_dir)
    decor_dir = out_dir / "decorations"
    inpaint_dir = decor_dir / "inpainted"
    objects_dir = decor_dir / "objects"
    results_path = decor_dir / "segment_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run inpaint_decorations first.\n"
            f"Expected: {results_path}"
        )

    with open(results_path) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[generate_decoration_3d] No segments — nothing to do.")
        return objects_dir

    if indices is not None:
        segments = [s for s in segments if s["seg_index"] in indices]

    objects_dir.mkdir(parents=True, exist_ok=True)

    # Hunyuan3D is required; there is no second backend.
    if not _hunyuan_available():
        import time as _time
        print("[generate_decoration_3d] Hunyuan3D server down — waiting for it…")
        for _ in range(160):  # up to ~40 min
            _time.sleep(15)
            if _hunyuan_available():
                print("[generate_decoration_3d] Hunyuan3D server is up — proceeding")
                break
        else:
            raise RuntimeError(
                "Hunyuan3D server never came up — no fallback backend is "
                "configured in this release."
            )
    print(f"[generate_decoration_3d] {len(segments)} decoration(s) | generator: Hunyuan3D")

    success = 0
    for seg in segments:
        seg_idx = seg["seg_index"]
        phrase  = seg.get("phrase", "decoration")
        safe    = phrase.replace(" ", "_")
        stem    = f"decor_{seg_idx:02d}_{safe}"
        glb_path = objects_dir / f"{stem}.glb"

        print(f"\n[generate_decoration_3d] {seg_idx:02d} '{phrase}'")

        # Source image: prefer inpainted, fall back to canvas
        src: Path | None = None
        inpaint_file = seg.get("inpaint_file")
        if inpaint_file and (inpaint_dir / inpaint_file).exists():
            src = inpaint_dir / inpaint_file
            print(f"  source: inpainted/{inpaint_file}")
        if src is None:
            canvas_file = seg.get("canvas_file")
            if canvas_file:
                candidate = decor_dir / "segmented" / canvas_file
                if candidate.exists():
                    src = candidate
                    print(f"  source: segmented/{canvas_file} (inpainted not found)")
        if src is None:
            print("  no source image — skipping")
            continue

        if glb_path.exists():
            print(f"  reusing existing {glb_path.name}")
            seg["glb_file"] = str(glb_path.relative_to(out_dir))
            success += 1
            continue

        if not src.exists():
            print(f"  source file missing — skipping idx={seg_idx}")
            continue

        gen_ok = False
        try:
            gen_ok = _generate_hunyuan(src, glb_path, texture=texture)
        except HunyuanUnmeshableError as e:
            print(f"  [hunyuan] {e}")
            new_src = _reinpaint_opaque(out_dir, seg_idx)
            if new_src is None:
                print(f"  [reinpaint] failed — skipping idx={seg_idx}")
                continue
            try:
                gen_ok = _generate_hunyuan(new_src, glb_path, texture=texture)
            except HunyuanUnmeshableError as e2:
                print(f"  [hunyuan] still unmeshable after re-inpaint: {e2}")
                gen_ok = False
        except Exception as e:
            print(f"  generation error — skipping idx={seg_idx}: {e}")
            continue

        if not gen_ok:
            print(f"  Hunyuan failed — skipping idx={seg_idx}")
            continue

        print(f"  saved → {glb_path}")
        seg["glb_file"] = str(glb_path.relative_to(out_dir))
        success += 1

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[generate_decoration_3d] Done. {success}/{len(segments)} GLBs → {objects_dir}/")
    return objects_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 3D GLB models for decoration objects."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output dir (contains decorations/segment_results.json)")
    ap.add_argument("--indices", nargs="+", type=int, default=None,
                    help="Segment indices to process (default: all)")
    ap.add_argument("--no-texture", action="store_true",
                    help="Disable texture generation (use when Hunyuan started without --enable_tex)")
    args = ap.parse_args()
    run(
        output_dir=args.output_dir,
        indices=args.indices,
        texture=not args.no_texture,
    )


if __name__ == "__main__":
    main()
