"""
main.py — HARMONY end-to-end pipeline (v2).

Runs the full chain from a reference photo to a rendered 3D room:

  Step 1  — floorplan
              VLM scene analysis → VGGT depth + camera alignment
              → wall mesh (walls.obj) → textures → base render (render_final.png)

  Step 2  — segment_wall_objects   LocateAnything→SAM2 segmentation of wall objects
  Step 3  — inpaint_wall_objects   Qwen image-edit completion of each segment
  Step 4  — rectify_windows        Perspective-correct window/door inpaints
  Step 5  — object_generation  Window planes + generated reconstructions
  Step 6  — wall_mounted_placement  Place wall-object GLBs, composite render

  Step 7  — segment_furniture      LocateAnything→SAM2 segmentation of furniture
  Step 8  — inpaint_furniture      Qwen image-edit cleanup of furniture crops
  Step 9  — furniture_placement_analysis  VLM size/position/orientation analysis
  Step 10 — furniture_object_generation   the 3D generator furniture GLBs
  Step 11 — furniture_placement    Place furniture GLBs, composite render

  Step 12 — detect_missing_decorations  VLM finds unplaced decoration items
  Step 13 — segment_decorations    LocateAnything→SAM2 segmentation
  Step 14 — inpaint_decorations    Qwen image-edit cleanup
  Step 15 — decoration_object_generation  Hunyuan3D → decoration GLBs
  Step 16 — place_decorations      Place decoration GLBs, final composite

  Step 17 — lighting               VLM lighting estimate → lightings/

CLI:
    python main.py --image data/indoor_images/office8.avif
    python main.py --image <path> --output outputs/<dir>
    python main.py --image <path> --rerun-dir outputs/<dir>
    python main.py --image <path> --stages floorplan furniture_placement
    python main.py --image <path> --skip lighting
"""
from __future__ import annotations

import argparse
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

from floorplan.pipeline import run_pipeline
import object_placement._vlm_instrument  # noqa: F401  (no-op unless SCENEWEAVE_VLM_LOG set)

# ── Stage registry ────────────────────────────────────────────────────────────

ALL_STAGES: list[str] = [
    "floorplan",
    "geometry_refine",
    "gen_empty_room",
    "texture_from_reference",
    "segment_wall_objects",
    "inpaint_wall_objects",
    "rectify_windows",
    "object_generation",
    "wall_mounted_materials",
    "wall_mounted_placement",
    "segment_furniture",
    "inpaint_furniture",
    "furniture_placement_analysis",
    "furniture_object_generation",
    "furniture_placement",
    "furniture_materials",
    "detect_missing_decorations",
    "segment_decorations",
    "inpaint_decorations",
    "decoration_object_generation",
    "place_decorations",
    "decoration_materials",
    # Ceiling fixtures composite LAST (hung from the ceiling, drawn on top of the
    # furniture+decoration render) — must run AFTER place_decorations so the
    # ceiling base render includes decorations and the latest furniture, not a
    # stale pre-decoration composite.
    "segment_ceiling_objects",
    "inpaint_ceiling_objects",
    "ceiling_object_generation",
    "ceiling_materials",
    "ceiling_placement",
    "lighting",
    "assemble_scene",
    # Whole-scene completeness audit, LAST — after every domain has had its
    # turn.  The per-stage `_recover_missed` hooks catch what a single
    # domain's detector dropped while that domain can still act on it (a
    # window recovered during the wall stage still becomes a light source).
    # This one is the backstop for everything that survived all of them, and
    # it looks at ANY type — furniture, wall-mounted and decoration alike.
    # Without it the pipeline has no step that compares the finished scene
    # against the photograph and asks what is simply absent: elegant shipped
    # without the foreground table that carried the room's main lamp, and
    # pexels_2343465 without the pedestal table between its two armchairs.
    "check_missing",
]

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HARMONY full pipeline — reference photo → 3D scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", required=True,
                   help="Reference indoor photo path.")
    p.add_argument("--output", default=None,
                   help="Output directory (default: outputs/<timestamp>/).")
    p.add_argument("--rerun-dir", default=None,
                   help="Existing output directory to resume from. "
                        "Mutually exclusive with --output.")
    p.add_argument("--stages", nargs="+", default=None, choices=ALL_STAGES,
                   metavar="STAGE",
                   help="Explicit list of stages to run (default: all).")
    p.add_argument("--skip", nargs="+", default=[], choices=ALL_STAGES,
                   metavar="STAGE",
                   help="Stages to skip.")
    p.add_argument("--locate-anything", action="store_true",
                   help="Deprecated no-op: LocateAnything→SAM2 (cross-env: "
                        "eagle-embodied + lang-sam) is now the only segmentation "
                        "backend for the wall/furniture/decoration stages. Kept for "
                        "backward-compatible command lines.")
    p.add_argument("--no-vlm", action="store_true",
                   help="Disable VLM post-checks in placement stages.")
    p.add_argument("--base-render", default=None,
                   help="Override background canvas for furniture compositing.")
    p.add_argument("--gemini-model", default="gemini-2.5-flash-image",
                   help="Gemini model for empty-room generation "
                        "(default: gemini-2.5-flash-image).")
    p.add_argument("--animate", action="store_true",
                   help="Save furniture/placement_animation.gif showing every "
                        "placement optimization step (scale/move/silhouette/depth).")
    p.add_argument("--animate-width", type=int, default=960,
                   help="Width of the placement animation GIF (default 960).")
    p.add_argument("--animate-frame-ms", type=int, default=1000,
                   help="Per-frame duration in ms for the animation (default 1000).")
    return p.parse_args()


args = _parse_args()
ref_image       = args.image
no_vlm          = args.no_vlm
# Segmentation backend: LocateAnything→SAM2 is now the only path. The
# wall/furniture/decoration segment stages always use
# object_placement.segmentation_la_sam2 (cross-env eagle-embodied + lang-sam);
# the legacy Grounded-SAM (GroundingDINO + SAM) fallback has been removed.
# --locate-anything / SCENEWEAVE_SEG_BACKEND are kept as deprecated no-ops.
print("[main] segmentation backend: LocateAnything→SAM2")
base_render     = args.base_render
gemini_model    = args.gemini_model
animate           = args.animate
animate_width     = args.animate_width
animate_frame_ms  = args.animate_frame_ms

stages = (list(args.stages) if args.stages
          else [s for s in ALL_STAGES if s not in set(args.skip)])

# ── Output directory ──────────────────────────────────────────────────────────

if args.rerun_dir:
    output_dir = Path(args.rerun_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"HARMONY — RERUN  image={ref_image}  dir={output_dir}")
elif args.output:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"HARMONY — image={ref_image}  dir={output_dir}")
else:
    output_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"HARMONY — image={ref_image}  dir={output_dir}")

# Store a canonical copy of the reference image inside the output dir so
# --rerun-dir is self-contained without needing --image again.
_src = Path(ref_image)
_dest = output_dir / (output_dir.name + _src.suffix)
if not _dest.exists() and _src.exists():
    shutil.copy2(_src, _dest)

# ── Stage runner ──────────────────────────────────────────────────────────────

results: dict = {}


def _banner(name: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {name.upper()}")
    print("═" * width)


# Outcome of every stage this run: name → "ok" | "failed" | "skipped:<why>".
# Previously _run_stage returned True/False and all 29 call sites DISCARDED it,
# so a failed stage printed a traceback and the pipeline carried on regardless
# — then printed "PIPELINE COMPLETE" at the end either way.  Four of the five
# July scenes finished with failed stages and every one of them reported
# success; living_room004 lost its whole decoration chain (segment → inpaint →
# object_gen → place → materials, five consecutive failures, each caused by the
# one before) and still reported complete.  Record what actually happened.
stage_outcomes: "dict[str, str]" = {}

# What each stage needs before it can do anything.  Paths are relative to
# output_dir; a stage whose input is missing is SKIPPED with the reason instead
# of being run so it can fail on the same missing file — that is what turned
# one real failure in living_room004 into a five-failure cascade.
STAGE_REQUIRES: "dict[str, list[str]]" = {
    "inpaint_wall_objects":        ["wall_mounted/segment_results.json"],
    "rectify_windows":             ["wall_mounted/segment_results.json"],
    "object_generation":     ["wall_mounted/segment_results.json"],
    "wall_mounted_materials":      ["wall_mounted/objects"],
    "wall_mounted_placement":      ["wall_mounted/objects"],
    "inpaint_furniture":           ["furniture/segment_results.json"],
    "furniture_placement_analysis": ["furniture/segment_results.json"],
    "furniture_object_generation": ["furniture/segment_results.json"],
    "furniture_placement":         ["furniture/objects"],
    "furniture_materials":         ["furniture/objects"],
    "segment_decorations":         ["decorations/decoration_analysis.json"],
    "inpaint_decorations":         ["decorations/segment_results.json"],
    "decoration_object_generation": ["decorations/segment_results.json"],
    "place_decorations":           ["decorations/objects",
                                    "furniture/furniture_placements.json"],
    "decoration_materials":        ["decorations/objects"],
    "inpaint_ceiling_objects":     ["ceiling/segment_results.json"],
    "ceiling_object_generation":   ["ceiling/segment_results.json"],
    "ceiling_materials":           ["ceiling/objects"],
    "ceiling_placement":           ["ceiling/objects"],
}


def _run_stage(name: str, fn) -> bool:
    """Run one stage. Returns True on success, False on skip/failure."""
    if name not in stages:
        print(f"[pipeline] Skipping '{name}'")
        stage_outcomes[name] = "skipped:not-requested"
        return False
    missing = [r for r in STAGE_REQUIRES.get(name, [])
               if not (output_dir / r).exists()]
    if missing:
        # Its producer failed or never ran.  Running anyway just reproduces the
        # same FileNotFoundError one stage later and buries the real cause.
        print(f"[pipeline] SKIPPING '{name}' — missing required input(s): "
              f"{', '.join(missing)}")
        stage_outcomes[name] = f"skipped:missing {missing[0]}"
        return False
    _banner(name)
    os.environ["SCENEWEAVE_STAGE"] = name
    _t_stage = datetime.now()
    try:
        fn()
        if os.environ.get("SCENEWEAVE_VLM_LOG"):
            print(f"[stage-time] {name}: {(datetime.now()-_t_stage).total_seconds():.1f}s")
        stage_outcomes[name] = "ok"
        return True
    except Exception as exc:
        print(f"[pipeline] Stage '{name}' FAILED: {exc}")
        traceback.print_exc()
        stage_outcomes[name] = f"failed:{type(exc).__name__}: {exc}"
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Floorplan
#   VLM scene analysis → VGGT depth + camera alignment
#   → wall mesh (walls.obj) → per-wall textures → base render
# ─────────────────────────────────────────────────────────────────────────────

def _stage_floorplan() -> None:
    results.update(run_pipeline(
        image_path=ref_image,
        output_dir=str(output_dir),
    ))


_run_stage("floorplan", _stage_floorplan)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — Depth-guided geometry refinement
#   Uses VGGT metric depth to extend each wall to its true measured extent
#   (under-constrained left/right walls otherwise stay length=0) and re-origins
#   the box so the camera is inside.  strip_openings=False keeps detected
#   windows/doors for the wall-mounted stages downstream.
# ─────────────────────────────────────────────────────────────────────────────

def _stage_geometry_refine() -> None:
    from floorplan.geometry_refine import refine_geometry
    try:
        rep = refine_geometry(str(output_dir), dry_run=False, use_depth=True,
                              strip_openings=False)
        results["geometry_refine"] = rep.get("actions")
        print(f"[geometry_refine] actions: {rep.get('actions')}")
        if rep.get("box_old") and rep.get("box_new_thresh"):
            print(f"[geometry_refine] box {rep['box_old']} → {rep['box_new_thresh']}")
    except Exception as exc:
        print(f"[geometry_refine] skipped — {type(exc).__name__}: {exc}")


_run_stage("geometry_refine", _stage_geometry_refine)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generate empty-room reference image
#   Uses render_final.png (geometry) + reference photo (texture/colour) →
#   calls Gemini/Qwen to remove all furniture → empty_room_ref.png
# ─────────────────────────────────────────────────────────────────────────────

def _stage_gen_empty_room() -> None:
    import os
    # If a background is already present in the scene folder (e.g. a GPT/clean
    # empty room copied in, or a previously-generated one), use it as-is and
    # skip regeneration. Deleting empty_room_ref.png forces a fresh generation.
    erp = output_dir / "empty_room_ref.png"
    if erp.exists():
        print(f"[gen_empty_room] empty_room_ref.png already present → "
              f"skipping generation, using existing background ({erp})")
        results["empty_room_ref"] = str(erp)
        return
    from floorplan.gen_empty_room_ref import process_scene
    backend = os.environ.get("SCENEWEAVE_IMG_EDIT", "qwen").lower()
    api_key = os.environ.get("GEMINI_API_KEY") if backend == "gemini" else None
    ok = process_scene(
        scene_dir=output_dir,
        backend=backend,
        model=gemini_model,
        api_key=api_key,
        skip_existing=False,
        ref_path=ref_image,      # pass directly so file discovery can't fail
    )
    if ok:
        results["empty_room_ref"] = str(output_dir / "empty_room_ref.png")


_run_stage("gen_empty_room", _stage_gen_empty_room)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Bake exact wall/floor textures from the empty-room reference
#   Uses empty_room_ref.png as the background for texture backprojection
# ─────────────────────────────────────────────────────────────────────────────

def _stage_texture_from_reference() -> None:
    from floorplan.texture_from_reference import run as run_ref_tex
    bg = results.get("empty_room_ref") or str(output_dir / "empty_room_ref.png")
    if not Path(bg).exists():
        print(f"[texture_from_reference] empty_room_ref.png not found — skipping")
        return
    r = run_ref_tex(output_dir=output_dir, bg_image_path=bg)
    results["ref_texture_metadata"] = r.get("metadata_path")
    results["ref_texture_render"]   = r.get("render_path")


_run_stage("texture_from_reference", _stage_texture_from_reference)

# ─────────────────────────────────────────────────────────────────────────────
# Steps 4–8 — Wall-mounted objects
# ─────────────────────────────────────────────────────────────────────────────

def _recover_missed(domain: str) -> None:
    """Per-stage safety net: ask the VLM for anything THIS domain's detector
    missed, and re-segment so the recovered objects join the contract.

    Scoped to one domain and run immediately after that domain's segmentation,
    because WHEN a miss is caught decides whether it can still matter.  A
    window recovered here becomes a light source for the lighting stage; the
    same window recovered by a global pass at the end arrives after lighting
    has already run with no sources and shipped a black render — which is
    exactly what happened to living_room8 and pexels_12881062.

    Off by default: set SCENEWEAVE_RECOVER_MISSED=1 (or =wall,furniture) to
    enable.  Never fatal — a failed recovery leaves the stage's own result.
    """
    flag = os.environ.get("SCENEWEAVE_RECOVER_MISSED", "")
    if not flag or flag == "0":
        return
    if flag not in ("1", "all") and domain not in {
            d.strip() for d in flag.split(",")}:
        return
    try:
        from object_placement import vlm_box_recovery as _vbr
        from object_placement.segmentation_la_sam2 import run_sam2_only
        if _vbr.recover(output_dir, ref_image, domain=domain) is not None:
            run_sam2_only(output_dir, domain)
    except Exception as exc:
        print(f"[recover:{domain}] skipped: {exc}")


def _stage_segment_wall_objects() -> None:
    from object_placement.segmentation_la_sam2 import run as run_seg
    results["wall_mounted_dir"] = str(run_seg(output_dir=output_dir,
                                              image_path=ref_image, domain="wall",
                                              no_verify=no_vlm))
    _recover_missed("wall")


_run_stage("segment_wall_objects", _stage_segment_wall_objects)


def _stage_inpaint_wall_objects() -> None:
    from object_placement.wall_mounted.inpaint_wall_objects_keep_glass import run as run_inpaint
    results["wall_mounted_inpainted_dir"] = str(run_inpaint(output_dir=output_dir))


_run_stage("inpaint_wall_objects", _stage_inpaint_wall_objects)


def _stage_rectify_windows() -> None:
    from object_placement.wall_mounted.rectify_windows import run as run_rectify
    run_rectify(output_dir=output_dir)


_run_stage("rectify_windows", _stage_rectify_windows)


def _stage_object_generation() -> None:
    from object_placement.wall_mounted.object_generation import run as run_object_gen
    results["wall_mounted_objects_dir"] = str(
        run_object_gen(wall_mounted_dir=output_dir / "wall_mounted")
    )


_run_stage("object_generation", _stage_object_generation)


def _stage_wall_mounted_materials() -> None:
    # VLM-estimated PBR materials (category / roughness / metallic / IOR) written
    # onto each wall-object GLB, so the assembled scene shades them correctly
    # (matte fabric curtains, metallic sconces, glass/painted frames).
    from object_placement.wall_mounted.material_properties import run as run_mat
    results["wall_mounted_materials"] = run_mat(output_dir=output_dir)


_run_stage("wall_mounted_materials", _stage_wall_mounted_materials)


def _stage_wall_mounted_placement() -> None:
    from object_placement.wall_mounted.placements.place_objects import (
        run_corner_based as run_place,
    )
    # Composite wall objects over the REFERENCE-textured render so the walls show
    # the real textures (render_ref_texture.png), not the plain floorplan textures
    # that render_room would draw from walls_metadata.json.  This textured base
    # then propagates to the furniture/ceiling composites (which chain off
    # render_objects_placed.png).  Falls back to whatever --base-render gives, or
    # the default plain render if the ref texture isn't present.
    _ref_tex = output_dir / "render_ref_texture.png"
    _wall_base = base_render or (str(_ref_tex) if _ref_tex.exists() else None)
    run_place(output_dir=output_dir, image_path=Path(ref_image),
              use_vlm=not no_vlm, proj_tex_image_path=_wall_base)
    results["wall_mounted_placements_dir"] = str(
        output_dir / "wall_mounted" / "placements"
    )


_run_stage("wall_mounted_placement", _stage_wall_mounted_placement)

# ─────────────────────────────────────────────────────────────────────────────
# Steps 7–11 — Furniture
# ─────────────────────────────────────────────────────────────────────────────

def _stage_segment_furniture() -> None:
    from object_placement.segmentation_la_sam2 import run as run_seg
    results["furniture_dir"] = str(run_seg(output_dir=output_dir,
                                           image_path=ref_image, domain="furniture",
                                           no_verify=no_vlm))

    _recover_missed("furniture")


_run_stage("segment_furniture", _stage_segment_furniture)


def _stage_inpaint_furniture() -> None:
    from object_placement.furniture.inpaint_furniture import run as run_inpaint
    results["furniture_inpainted_dir"] = str(run_inpaint(output_dir=output_dir))


_run_stage("inpaint_furniture", _stage_inpaint_furniture)


def _stage_furniture_placement_analysis() -> None:
    from object_placement.furniture.object_placement import run as run_analysis
    results["furniture_placement_analysis"] = str(run_analysis(output_dir=output_dir))


_run_stage("furniture_placement_analysis", _stage_furniture_placement_analysis)


def _stage_furniture_object_generation() -> None:
    from object_placement.furniture.object_generation import run as run_object_gen
    results["furniture_objects_dir"] = str(
        run_object_gen(furniture_dir=output_dir / "furniture")
    )


_run_stage("furniture_object_generation", _stage_furniture_object_generation)


def _stage_furniture_placement() -> None:
    from object_placement.furniture.place_furniture_vggt import run as run_place
    results["furniture_placements_dir"] = str(
        run_place(output_dir=output_dir, use_vlm=not no_vlm,
                  base_image_path=base_render,
                  animate=animate, animate_gif_width=animate_width,
                  animate_frame_ms=animate_frame_ms)
    )


_run_stage("furniture_placement", _stage_furniture_placement)


def _stage_furniture_materials() -> None:
    # POST-placement finishing pass (placement-first is easier to debug): VLM
    # materials (category / roughness / metallic / IOR / transmission) written
    # onto each furniture GLB + physical props (weight / thickness / elasticity)
    # to a sidecar, THEN scene_with_furniture.glb is cheaply re-baked from the
    # saved placement JSON so the assembled scene shades correctly (matte fabric
    # sofas, glossy metal legs, transmissive glass tables).  No placement
    # re-compute — just re-reads the materialled object GLBs + transforms.
    from object_placement.furniture.material_properties import run as run_fmat
    results["furniture_materials"] = run_fmat(output_dir=output_dir,
                                              rebake_scene=True)


_run_stage("furniture_materials", _stage_furniture_materials)

# ─────────────────────────────────────────────────────────────────────────────
# Steps 12–16 — Decorations
# ─────────────────────────────────────────────────────────────────────────────

def _stage_detect_missing_decorations() -> None:
    from object_placement.decorations.detect_missing_decorations import run as run_detect
    results["decorations_detected"] = str(
        run_detect(output_dir=output_dir, image_path=ref_image)
    )


_run_stage("detect_missing_decorations", _stage_detect_missing_decorations)


def _stage_segment_decorations() -> None:
    from object_placement.segmentation_la_sam2 import run as run_seg
    results["decorations_dir"] = str(run_seg(output_dir=output_dir,
                                             image_path=ref_image, domain="decoration",
                                             no_verify=no_vlm))
    _recover_missed("decoration")


_run_stage("segment_decorations", _stage_segment_decorations)


def _stage_inpaint_decorations() -> None:
    from object_placement.decorations.inpaint_decorations import run as run_inpaint
    results["decorations_inpainted_dir"] = str(run_inpaint(output_dir=output_dir))


_run_stage("inpaint_decorations", _stage_inpaint_decorations)


def _stage_decoration_object_generation() -> None:
    from object_placement.decorations.generate_decoration_3d import run as run_3d
    results["decoration_objects_dir"] = str(
        run_3d(output_dir=output_dir)
    )


_run_stage("decoration_object_generation", _stage_decoration_object_generation)


def _stage_place_decorations() -> None:
    from object_placement.decorations.place_decorations import run as run_place
    results["decoration_placements_dir"] = str(
        run_place(output_dir=output_dir, use_vlm=not no_vlm)
    )


_run_stage("place_decorations", _stage_place_decorations)


def _stage_decoration_materials() -> None:
    # POST-placement finishing pass (mirrors furniture_materials): VLM materials
    # written onto each decoration GLB (matte fabric pillows, paper books,
    # ceramic bowl, glass lantern), then scene_with_decoration.glb is re-baked
    # from the saved decoration placement JSON — no re-placement.
    from object_placement.decorations.material_properties import run as run_dmat
    results["decoration_materials"] = run_dmat(output_dir=output_dir,
                                               rebake_scene=True)


_run_stage("decoration_materials", _stage_decoration_materials)

# ─────────────────────────────────────────────────────────────────────────────
# Ceiling fixtures — pendant lights / chandeliers / ceiling fans
#   Mirrors the wall-mounted path (segment → inpaint → reconstruct → place) but
#   anchors each fixture to the ceiling plane and composites over the furnished
#   render.  Runs LAST (after furniture + decorations) so the ceiling base render
#   includes decorations and the pendant hangs over the fully-placed scene.
# ─────────────────────────────────────────────────────────────────────────────

def _stage_segment_ceiling_objects() -> None:
    from object_placement.ceiling.segment_ceiling_objects import run as run_seg
    results["ceiling_dir"] = str(run_seg(output_dir=output_dir, image_path=ref_image))


_run_stage("segment_ceiling_objects", _stage_segment_ceiling_objects)


def _stage_inpaint_ceiling_objects() -> None:
    from object_placement.ceiling.inpaint_ceiling_objects import run as run_inpaint
    results["ceiling_inpainted_dir"] = str(run_inpaint(output_dir=output_dir))


_run_stage("inpaint_ceiling_objects", _stage_inpaint_ceiling_objects)


def _stage_ceiling_object_generation() -> None:
    # Reuse the wall-mounted Hunyuan3D generator — it is generic over the
    # folder it's pointed at (reads segment_results.json + inpainted/ → objects/).
    from object_placement.wall_mounted.object_generation import run as run_object_gen
    results["ceiling_objects_dir"] = str(
        run_object_gen(wall_mounted_dir=output_dir / "ceiling")
    )


_run_stage("ceiling_object_generation", _stage_ceiling_object_generation)


def _stage_ceiling_materials() -> None:
    # VLM materials onto each ceiling fixture GLB (glossy metal track light,
    # glass diffuser) BEFORE placement, so the ceiling composite shades it
    # correctly.  Texture is already baked from the inpaint; this adds the PBR
    # factors (metallic/roughness/IOR/transmission).
    from object_placement.ceiling.material_properties import run as run_cmat
    results["ceiling_materials"] = run_cmat(output_dir=output_dir)


_run_stage("ceiling_materials", _stage_ceiling_materials)


def _stage_ceiling_placement() -> None:
    from object_placement.ceiling.place_ceiling_objects import run as run_place
    results["ceiling_placements_dir"] = str(
        run_place(output_dir=output_dir, image_path=ref_image,
                  use_vlm=not no_vlm, base_image_path=base_render)
    )


_run_stage("ceiling_placement", _stage_ceiling_placement)

# ─────────────────────────────────────────────────────────────────────────────
# Step 17 — Lighting
# ─────────────────────────────────────────────────────────────────────────────

def _stage_lighting() -> None:
    from lighting_module.pipeline import run as run_lighting
    results["lightings_dir"] = str(
        run_lighting(output_dir=output_dir, image_path=ref_image)
    )
    # The Cycles render is part of the lighting phase: bake the carpet (2D
    # overlay → floor quad) so the rug appears in the path-traced scene, then
    # render with Blender.  Skipped gracefully if Blender isn't available.
    try:
        import subprocess as _sp, sys as _sys
        _sp.run([_sys.executable, "lighting_module/bake_carpet.py", "--scene",
                 str(output_dir)], check=False)
    except Exception as _e:
        print(f"[lighting] carpet bake skipped: {_e}")
    try:
        from lighting_module import render_blender
        out_png = render_blender.render(output_dir)
        results["render_blender"] = str(out_png)
        print(f"[lighting] Blender Cycles render → {out_png}")
    except Exception as _e:
        print(f"[lighting] Blender render skipped: {_e}")


_run_stage("lighting", _stage_lighting)

# ─────────────────────────────────────────────────────────────────────────────
# Final — assemble the whole scene into one textured GLB
#   Merges the textured room shell (walls/floor/ceiling) with every placed
#   object scene (furniture / ceiling / wall-mounted) into scene_full.glb.
# ─────────────────────────────────────────────────────────────────────────────

def _stage_assemble_scene() -> None:
    from object_placement.assemble_scene_glb import run as run_assemble
    results["scene_full_glb"] = str(run_assemble(output_dir=output_dir))


_run_stage("assemble_scene", _stage_assemble_scene)


# ─────────────────────────────────────────────────────────────────────────────
# Final — whole-scene completeness audit (all object types)
# ─────────────────────────────────────────────────────────────────────────────

def _stage_check_missing() -> None:
    """Compare the finished scene against the reference and recover anything
    still absent, of any type.

    Report-only by default (SCENEWEAVE_CHECK_MISSING=report): it lists what is
    missing and writes missing_report.json, without generating meshes.  Set
    SCENEWEAVE_CHECK_MISSING=recover to go on and segment/generate/place them,
    which needs the Hunyuan server.
    """
    mode = os.environ.get("SCENEWEAVE_CHECK_MISSING", "report").lower()
    if mode in ("0", "off", "skip"):
        print("[check_missing] disabled (SCENEWEAVE_CHECK_MISSING=off)")
        return
    from object_placement.missing_items import place_missing_items as _pmi
    from PIL import Image as _Im

    # Use the module's own resolver so this agrees with what the recovery path
    # would load — falling back to --image would diverge on a --rerun-dir.
    try:
        ref = _pmi._resolve_ref_photo(output_dir, _pmi._load_camera(output_dir))
    except Exception as _re:
        print(f"[check_missing] could not resolve reference ({_re}) — "
              f"using {ref_image}")
        ref = Path(ref_image)
    render = None
    for cand in ("lightings/render_blender.png",
                 "decorations/placements/render_decorations_placed.png",
                 "furniture/render_furniture_placed.png"):
        if (output_dir / cand).exists():
            render = output_dir / cand
            break
    if render is None:
        print("[check_missing] no scene render to compare against — skipping")
        return
    print(f"[check_missing] comparing {Path(ref).name} against {render.name}")
    found = _pmi._vlm_detect_missing(_Im.open(ref).convert("RGB"),
                                     _Im.open(render).convert("RGB"),
                                     _pmi._placed_categories(output_dir))
    report = {"render": str(render), "missing": found}
    (output_dir / "missing_report.json").write_text(
        __import__("json").dumps(report, indent=2))
    if not found:
        print("[check_missing] nothing missing — scene matches the reference")
        return
    print(f"[check_missing] ⚠ {len(found)} object(s) in the photo are absent "
          f"from the scene:")
    for m in found:
        print(f"    {m['type']:<18} ({m.get('domain')}) "
              f"box={m.get('box_px')} conf={m.get('confidence', 0):.2f} "
              f"— {m.get('description', '')[:70]}")
    results["missing_report"] = str(output_dir / "missing_report.json")
    if mode == "recover":
        print("[check_missing] recovering …")
        _pmi.run(output_dir=output_dir, missing_types=None)


_run_stage("check_missing", _stage_check_missing)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

_ok      = [n for n, v in stage_outcomes.items() if v == "ok"]
_failed  = [(n, v) for n, v in stage_outcomes.items() if v.startswith("failed")]
_skipped = [(n, v) for n, v in stage_outcomes.items()
            if v.startswith("skipped") and v != "skipped:not-requested"]

print("\n" + "═" * 70)
if _failed or _skipped:
    # NOT "COMPLETE": reaching the last stage is not the same as succeeding.
    # Without this a batch of scenes reports success while stages fail inside,
    # and the only trace is a line buried in a per-scene log.
    print(f"  PIPELINE FINISHED WITH PROBLEMS — "
          f"{len(_ok)} ok, {len(_failed)} failed, {len(_skipped)} skipped")
else:
    print("  PIPELINE COMPLETE")
print("═" * 70)
print(f"  output_dir : {output_dir}")
for _n, _v in _failed:
    print(f"  ✗ FAILED  {_n}: {_v.split(':', 1)[1][:110]}")
for _n, _v in _skipped:
    print(f"  ⊘ SKIPPED {_n}: {_v.split(':', 1)[1][:110]}")
for k, v in results.items():
    if k != "analysis":
        print(f"  {k}: {v}")
try:
    (output_dir / "stage_outcomes.json").write_text(
        __import__("json").dumps(stage_outcomes, indent=2))
except Exception:
    pass
if _failed:
    # Non-zero exit so a batch driver can tell a broken scene from a good one.
    # Set SCENEWEAVE_TOLERATE_STAGE_FAILURE=1 to keep the old always-zero exit.
    if os.environ.get("SCENEWEAVE_TOLERATE_STAGE_FAILURE") != "1":
        raise SystemExit(1)
