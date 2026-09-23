"""
object_generation.py — generate 3D models for wall-mounted objects via
Hunyuan3D (the only 3D-generation backend in this release; the module was
in this release — Hunyuan3D is now required, not optional).

Reads segment_results.json from a wall_mounted output folder.  For each segment,
uses its inpainted image (from inpainted/) if available, otherwise the canvas PNG
(from segmented/).

On a Hunyuan3D 404 (the known can't-mesh-this signature — glass/transparent
objects deadlock marching cubes), this re-inpaints the segment with an
opaque-material qualifier appended to its detection phrase and retries once.

Outputs are saved to <wall_mounted_dir>/objects/inpaint_<idx>_<type>.glb.

Usage:
    python -m object_placement.wall_mounted.object_generation \\
        --wall-mounted-dir outputs/20260331_031530/wall_mounted
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Re-use Hunyuan3D helpers from the furniture pipeline
from object_placement.furniture.object_generation import (
    HunyuanUnmeshableError,
    _generate_hunyuan,
    _hunyuan_available,
)


def _reinpaint_opaque(output_dir: Path, idx: int) -> Path | None:
    """Re-inpaint one wall-mounted segment with an opaque-material qualifier
    and return its new image path, or None on failure."""
    from object_placement.wall_mounted.inpaint_wall_objects_keep_glass import (
        run as run_inpaint,
    )

    wm_dir    = output_dir / "wall_mounted"
    results_p = wm_dir / "segment_results.json"
    with open(results_p) as f:
        data = json.load(f)
    for s in data["segments"]:
        if s["index"] == idx:
            base = s.get("phrase", s.get("type", "object"))
            s["phrase"] = (
                f"{base}, made of solid opaque material (NOT glass, NOT "
                "transparent, NOT translucent)"
            )
            break
    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [reinpaint] regenerating idx={idx} as opaque material …")
    run_inpaint(output_dir, indices=[idx])

    with open(results_p) as f:
        data = json.load(f)
    for s in data["segments"]:
        if s["index"] == idx and s.get("inpaint_file"):
            return wm_dir / "inpainted" / s["inpaint_file"]
    return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

# Types built as thin textured plane meshes instead of a generated mesh.
#   - window / door / frame : always flat architectural elements
#   - art / painting        : flat 2D canvases / prints / drawings.  Non-flat
#     wall art (crochet, tapestry, sculpture) should be detected as "other"
#     (or another non-planar type) so it falls through to 3D generation.
_WINDOW_PLANE_TYPES = {"window", "door", "art", "painting", "frame"}


def run(wall_mounted_dir: str | Path,
        types: list[str] | None = None, force: bool = False,
        device: str = "cuda") -> Path:
    wm_dir      = Path(wall_mounted_dir)
    results_p   = wm_dir / "segment_results.json"
    inpaint_dir = wm_dir / "inpainted"
    seg_dir     = wm_dir / "segmented"
    out_dir     = wm_dir / "objects"

    if not results_p.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_wall_objects first.\n"
            f"Expected: {results_p}"
        )

    with open(results_p) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[object_gen] No segments — nothing to do.")
        return out_dir

    out_dir.mkdir(exist_ok=True)

    print(f"[object_gen] {len(segments)} segment(s) to process.")

    # Hunyuan3D is required; there is no second backend.
    if not _hunyuan_available():
        import time as _time
        print("[object_gen] Hunyuan3D server down — waiting for it…")
        for _ in range(160):  # up to ~40 min
            _time.sleep(15)
            if _hunyuan_available():
                print("[object_gen] Hunyuan3D server is up — proceeding")
                break
        else:
            raise RuntimeError(
                "Hunyuan3D server never came up — no fallback backend is "
                "configured in this release."
            )
    print("[object_gen] Generator: Hunyuan3D")

    success = 0
    for seg in segments:
        idx      = seg["index"]
        obj_type = seg.get("type", "other")
        stem     = f"inpaint_{idx:02d}_{obj_type}"
        glb_path = out_dir / f"{stem}.glb"

        if types is not None and obj_type not in types:
            continue

        print(f"\n[object_gen] {idx:02d} {obj_type}")

        # Prefer inpainted image; fall back to canvas
        src: Path | None = None
        if seg.get("inpaint_file"):
            candidate = inpaint_dir / seg["inpaint_file"]
            if candidate.exists():
                src = candidate
                print(f"  using inpainted: {candidate.name}")
        if src is None and seg.get("canvas_file"):
            candidate = seg_dir / seg["canvas_file"]
            if candidate.exists():
                src = candidate
                print(f"  inpainted not found — using canvas: {candidate.name}")
        if src is None:
            print("  no source image found — skipping.")
            continue

        if glb_path.exists() and not force:
            print(f"  reusing existing {glb_path.name}")
            seg["glb_file"] = str(glb_path.relative_to(wm_dir))
            success += 1
            continue

        if obj_type in _WINDOW_PLANE_TYPES:
            from object_placement.wall_mounted.generate_window_plane import make_window_plane
            from PIL import Image as _Image
            # Rectify every plane type: windows/doors are often captured at an
            # angle (perspective trapezoid); art/painting/frame need the same
            # fronto-parallel correction.
            _RECTIFY_TYPES = {"window", "door", "art", "painting", "frame"}
            rectify = obj_type in _RECTIFY_TYPES
            # SCENEWEAVE_NO_WINDOW_RECTIFY=1 keeps curtained/soft windows as a
            # flat plane straight from the segmentation instead of warping them
            # fronto-parallel (which turns a curtain into a narrow "frame" strip).
            if os.environ.get("SCENEWEAVE_NO_WINDOW_RECTIFY") and obj_type in ("window", "curtain"):
                rectify = False
            try:
                img     = _Image.open(src).convert("RGB")
                glb_bytes = make_window_plane(img, device=device,
                                              rectify=rectify)
                glb_path.write_bytes(glb_bytes)
                print(f"  [plane] saved → {glb_path}")
                seg["glb_file"] = str(glb_path.relative_to(wm_dir))
                success += 1
            except Exception as e:
                print(f"  [plane] FAILED: {e}")
            continue

        # 3D path (mirror, shelf, cabinet, light, …) via Hunyuan3D. On the
        # known unmeshable-object 404, re-inpaint as opaque and retry once.
        try:
            gen_ok = _generate_hunyuan(src, glb_path)
        except HunyuanUnmeshableError as e:
            print(f"  [hunyuan] {e}")
            new_src = _reinpaint_opaque(wm_dir.parent, idx)
            if new_src is None:
                print(f"  [reinpaint] failed — skipping idx={idx}")
                continue
            try:
                gen_ok = _generate_hunyuan(new_src, glb_path)
            except HunyuanUnmeshableError as e2:
                print(f"  [hunyuan] still unmeshable after re-inpaint: {e2}")
                gen_ok = False

        if not gen_ok:
            print(f"  Hunyuan failed — skipping idx={idx}")
            continue
        print(f"  saved → {glb_path}")
        seg["glb_file"] = str(glb_path.relative_to(wm_dir))
        success += 1

    # Persist glb_file paths back into segment_results.json
    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[object_gen] Done. {success}/{len(segments)} models → {out_dir}/")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 3D GLB models for wall-mounted objects via Hunyuan3D."
    )
    ap.add_argument(
        "--wall-mounted-dir", required=True,
        help="Path to the wall_mounted output folder (contains segment_results.json)",
    )
    ap.add_argument(
        "--types", default=None,
        help="Comma-separated object types to generate (e.g. window,door). Default: all.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate even if GLB already exists.",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Device for window-plane SAM segmentation (cuda or cpu, default: cuda)",
    )
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    run(wall_mounted_dir=args.wall_mounted_dir,
        types=types, force=args.force, device=args.device)


if __name__ == "__main__":
    main()
