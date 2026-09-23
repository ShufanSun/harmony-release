"""Lighting stage orchestrator.

Outputs are written under `<output_dir>/lightings/`:
  lights.json                 — final lighting setup (ambient + directional +
                                point + window daylight) keyed by source id
  sources.json                — light-source records collected from earlier stages
  objects/                    — emissive GLB copies (lit_<name>.glb)
  render_lit_iter_0.png       — preview after the initial VLM estimate
  render_lit_iter_<N>.png     — preview after each refinement step
  render_lit.png              — final preview (alias of last iteration)
  refine_log.json             — VLM verdicts + deltas applied per iteration

Other stages are never modified.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import analyze, emissive, light_sources, refine, render_lit, render_pyrender


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _resolve_reference_image(out_dir: Path,
                             supplied: str | Path | None) -> Path:
    """Return the reference photo. Falls back to the first image-like file at
    the top of the run dir (re-runs typically already have the original photo
    copied in, e.g. <out>/<out.name>.jpg)."""
    if supplied:
        p = Path(supplied)
        if p.exists():
            return p

    # Common fallbacks: <out>/<out.name>.{jpg,jpeg,png,avif} then any image
    # directly inside the output dir.
    stem = out_dir.name
    for ext in ("jpg", "jpeg", "png", "avif", "JPG", "JPEG", "PNG"):
        cand = out_dir / f"{stem}.{ext}"
        if cand.exists():
            return cand
    for cand in sorted(out_dir.glob("*")):
        if cand.is_file() and cand.suffix.lower() in {".jpg", ".jpeg", ".png", ".avif"}:
            return cand

    raise FileNotFoundError(
        f"Reference image not found. Tried '{supplied}' and the top of "
        f"'{out_dir}' — pass --image with a real path."
    )


def run(output_dir: str | Path, image_path: str | Path | None = None,
        n_iterations: int | None = None) -> Path:
    """Run the full lighting stage. Returns the lightings directory."""
    # Default raised from 2 to 4: with 2, a scene needing three corrections ran
    # out of iterations mid-way and shipped a half-corrected rig without saying
    # so (pexels_2343465 — see the non-convergence warning below).  The loop
    # still exits as soon as the VLM says "converged", so well-behaved scenes
    # cost nothing extra.
    if n_iterations is None:
        n_iterations = int(os.environ.get("SCENEWEAVE_LIGHT_ITERS", "4"))
    out_dir = Path(output_dir)
    light_dir = out_dir / "lightings"
    objects_dir = light_dir / "objects"
    light_dir.mkdir(parents=True, exist_ok=True)
    objects_dir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_reference_image(out_dir, image_path)
    print(f"[lighting] using reference image: {image_path}")

    # 1. Collect light-source records
    sources = light_sources.collect(out_dir)
    if not sources:
        # A room with NO collectable light source is nearly always an upstream
        # detection miss, not a genuinely unlit room: the wall-mounted stage
        # found no window/lamp, so nothing reached light_sources.collect().
        # The old fallback wrote ambient 0.5 and returned, which left
        # render_blender with lights=[] and a world of 0.6 × 0.8 — pitch black
        # (living_room8, pexels_12881062 both shipped that way), and because
        # the refine loop never ran there was no iteration to catch it.
        #
        # So: say plainly what is wrong and where to look, and write a rig that
        # at least RENDERS. This is a visible fallback, not a correct one — the
        # real fix is upstream detection.
        wm = out_dir / "wall_mounted" / "segment_results.json"
        detail = ""
        try:
            _seg = json.loads(wm.read_text())
            if not _seg.get("segments"):
                detail = (f" — wall_mounted detected NOTHING "
                          f"(absent: {', '.join(_seg.get('absent', [])[:6])}…), "
                          f"so there is no window or lamp to light the room")
        except Exception:
            pass
        print(f"[lighting] ⚠ NO LIGHT SOURCES{detail}")
        print(f"[lighting] ⚠ falling back to a flat ambient fill so the Blender "
              f"render is not black; brightness will NOT match the reference "
              f"and no refinement iterations will run")
        _atomic_write_json(light_dir / "lights.json",
                           {"ambient": {"color": [1, 1, 1], "intensity": 0.5},
                            "directional": [], "point_lights": [], "windows": [],
                            # base_dim drives world strength in render_blender's
                            # _world_from_ambient; without it the default 0.5
                            # combined with no lights renders near-black.
                            "base_dim": float(os.environ.get(
                                "SCENEWEAVE_NOLIGHT_BASE_DIM", "1.15")),
                            "_fallback_no_sources": True})
        return light_dir

    print(f"[lighting] collected {len(sources)} light source(s):")
    for s in sources:
        print(f"  {s['kind']:16s}  {s['phrase']:24s}  "
              f"pos={tuple(round(v, 2) for v in s['position_m'])}")

    # 2. VLM initial estimate (mutates `sources` in place to assign ids)
    lights = analyze.estimate(out_dir, image_path, sources)
    _atomic_write_json(light_dir / "sources.json",
                       [{k: v for k, v in s.items() if k != "raw"}
                        for s in sources])
    _atomic_write_json(light_dir / "lights.json", lights)

    # 3. Edit emissive materials on the lit objects
    edits = emissive.edit_all(sources, lights, objects_dir)
    _atomic_write_json(light_dir / "emissive_edits.json", edits)
    print(f"[lighting] wrote {len(edits)} emissive GLB(s) → {objects_dir}")

    # 4. Initial preview — fast 2D ellipse-shadow preview
    preview_path = light_dir / "render_lit_iter_0.png"
    render_lit.render(out_dir, lights, sources, preview_path)

    # Real shadow render via pyrender (offscreen). Saved next to the 2D
    # preview; non-fatal if the GL context can't be created.
    pyrender_path = light_dir / "render_pyrender_iter_0.png"
    render_pyrender.render(out_dir, lights, sources, pyrender_path)

    # 5. VLM refinement loop
    refine_log: list[dict] = []
    for it in range(1, max(1, int(n_iterations)) + 1):
        if not preview_path.exists():
            print("[lighting] no preview to refine — stopping")
            break
        print(f"[lighting] refinement iteration {it}/{n_iterations}")
        deltas = refine.request_deltas(image_path, preview_path, lights, sources)
        if deltas is None:
            print("[lighting] VLM returned no deltas — stopping refinement")
            break

        verdict = (deltas.get("verdict") or "").lower()
        brightness = deltas.get("brightness_assessment", "?")
        tone = deltas.get("tone_assessment", "?")
        g_mult = deltas.get("global_intensity_mult", 1.0)
        print(f"[lighting] VLM iter {it}: brightness={brightness}  tone={tone}  "
              f"global_mult={g_mult}  verdict={verdict!r}")

        lights = refine.apply_deltas(lights, deltas)
        _atomic_write_json(light_dir / "lights.json", lights)

        new_preview = light_dir / f"render_lit_iter_{it}.png"
        render_lit.render(out_dir, lights, sources, new_preview)
        render_pyrender.render(out_dir, lights, sources,
                               light_dir / f"render_pyrender_iter_{it}.png")
        refine_log.append({
            "iteration": it,
            "brightness_assessment": brightness,
            "tone_assessment": tone,
            "global_intensity_mult": g_mult,
            "verdict": verdict,
            "deltas": deltas,
        })
        _atomic_write_json(light_dir / "refine_log.json", refine_log)

        preview_path = new_preview
        if "converged" in verdict:
            print("[lighting] VLM reports convergence — stopping early")
            break
    else:
        # Ran out of iterations with the VLM still asking for changes.  The old
        # loop exited here silently and shipped a half-corrected rig, which is
        # how pexels_2343465 finished: both iterations returned
        # brightness=too_bright tone=too_cool verdict=closer, the counter hit 2,
        # and the render went out still too bright and too cool.  A refinement
        # loop that stops while its critic is still objecting has not refined
        # anything — say so, and record it next to the result.
        if refine_log:
            _last = refine_log[-1]
            print(f"[lighting] ⚠ DID NOT CONVERGE after {n_iterations} "
                  f"iteration(s) — VLM still reports "
                  f"brightness={_last.get('brightness_assessment')} "
                  f"tone={_last.get('tone_assessment')} "
                  f"(verdict={_last.get('verdict')!r}). The rig shipped here is "
                  f"partially corrected; raise SCENEWEAVE_LIGHT_ITERS to give "
                  f"the loop room to finish.")
            _last["converged"] = False
            _atomic_write_json(light_dir / "refine_log.json", refine_log)

    # 6. Alias the last preview as render_lit.png
    if preview_path.exists():
        final = light_dir / "render_lit.png"
        try:
            shutil.copy2(preview_path, final)
            print(f"[lighting] final preview → {final}")
        except Exception as e:
            print(f"[lighting] failed to alias final preview: {e}")

    print(f"[lighting] done — outputs in {light_dir}")
    return light_dir


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Estimate scene lighting via VLM.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image", default=None,
                    help="Reference photo. If omitted, the first image at the "
                         "top of --output-dir is used.")
    ap.add_argument("--iterations", type=int, default=2)
    a = ap.parse_args()
    run(a.output_dir, a.image, n_iterations=a.iterations)


if __name__ == "__main__":
    main()
