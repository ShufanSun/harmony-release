"""
gen_empty_room_ref.py — generate a furniture-free empty-room image from the
reference photo, guided by render_final.png, for use in texture backprojection.

Uses two images as guidance:
  1. render_final.png  — clean 3-D reconstruction (structural guidance:
                          correct geometry, perspective, wall/floor layout)
  2. reference photo   — real photo (texture/colour guidance: real wall
                          paint, floor material, natural lighting)

Backend selection (--backend / SCENEWEAVE_IMG_EDIT env var):
  qwen   — local Qwen img_server (default).  The reference photo is passed
            directly; the prompt handles object removal with low cfg_scale
            to preserve real surface textures.
  gemini — Gemini image API.  Both images are sent as separate parts in one
            request (render_final.png used as geometric guidance).

Output: <scene_dir>/empty_room_ref.png

Setup (Qwen):
    # Start the server first:
    cd Qwen && uvicorn img_server:app --port 8000
    python floorplan/gen_empty_room_ref.py --scene outputs/front3d/rgb_006006

Setup (Gemini):
    export GEMINI_API_KEY="AIza..."
    python floorplan/gen_empty_room_ref.py --scene outputs/front3d/rgb_006006 --backend gemini

Usage:
    # Single scene (Qwen):
    python floorplan/gen_empty_room_ref.py --scene outputs/front3d/rgb_006006

    # All scenes (Gemini):
    python floorplan/gen_empty_room_ref.py --root outputs/front3d --backend gemini --skip-existing

    # Specific scenes:
    python floorplan/gen_empty_room_ref.py --root outputs/front3d \\
        --only rgb_006006 rgb_004818 --backend qwen
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_NAME = "empty_room_ref.png"


# ── prompts ───────────────────────────────────────────────────────────────────

# Pass 1 — aggressive removal (high cfg_scale): erase every object, don't
# worry about perfect surface fill quality yet.
PROMPT_REMOVE = (
    "Remove EVERY object from this room without exception. "
    "This includes — but is not limited to — "
    "sofas, chairs, armchairs, stools, benches, "
    "desks, tables, coffee tables, side tables, "
    "beds, wardrobes, dressers, shelves, bookcases, cabinets, "
    "ALL plants: potted plants, floor plants, hanging plants, any greenery whatsoever, "
    "ALL ceiling lights: pendant lamps, chandeliers, track lights, ceiling fixtures, "
    "  recessed lights, hanging bulbs — remove completely and fill with bare ceiling, "
    "floor lamps, table lamps, wall sconces, "
    "rugs, carpets, mats, "
    "curtains, blinds, drapes — remove and fill window opening with surrounding wall, "
    "artwork, mirrors, clocks, wall shelves, TVs, "
    "coat hangers, clothes racks, fans, radiators, speakers, bins, boxes. "
    "The output MUST contain zero objects — only bare walls, bare floor, bare ceiling. "
    "Fill every removed region with the surrounding surface colour."
)

# Pass 2 — inpaint cleanup (low cfg_scale): smooth and blend the filled regions,
# restore texture continuity, and lock untouched surfaces to the original.
PROMPT_INPAINT = (
    "This is an almost-empty room. Seamlessly inpaint any patchy, smeared, or "
    "inconsistent regions on the walls, floor, and ceiling so the surface looks "
    "continuous and natural. "
    "CRITICAL: every pixel that already looks clean and correct must remain "
    "absolutely identical — same colour, texture grain, shadows, ambient occlusion, "
    "and tonal gradients as the input. Only blend the uneven filled patches."
)

# Gemini gets the render as a second image for geometric guidance where
# furniture occludes the walls/floor.
PROMPT_GEMINI = (
    "You are given two images of the same room:\n"
    "  Image 1: the real reference photograph.\n"
    "  Image 2: a clean 3-D render of the same empty room (use as geometric "
    "reference for wall/floor/ceiling surfaces hidden behind objects).\n\n"
    "Produce a completely empty room from Image 1. Remove and fill ALL of:\n"
    "• Every piece of furniture: sofas, chairs, tables, desks, beds, wardrobes, "
    "shelves, cabinets, stools\n"
    "• ALL plants: potted plants, floor plants, hanging plants, any greenery\n"
    "• ALL ceiling lights: pendant lamps, chandeliers, ceiling fixtures, track "
    "lights, hanging bulbs — replace with bare ceiling surface\n"
    "• ALL floor lamps, table lamps, wall sconces\n"
    "• ALL rugs, carpets, mats\n"
    "• ALL curtains, blinds, drapes — remove and fill the window opening with "
    "the surrounding wall material so the wall looks continuous\n"
    "• ALL wall-mounted objects: artwork, mirrors, clocks, TVs, shelves\n"
    "• ALL other freestanding or hanging objects\n\n"
    "Fill every removed region seamlessly using the real surface texture visible "
    "in Image 1. Use Image 2 only as geometric guidance for surface edges "
    "occluded by objects.\n"
    "Output: a single image — bare walls, bare floor, bare ceiling, zero objects."
)

NEG_PROMPT = (
    "furniture, chairs, sofas, tables, desks, beds, shelves, lamps, plants, "
    "potted plants, ceiling lamp, pendant light, chandelier, hanging light, "
    "rugs, carpets, artwork, mirrors, curtains, blinds, clocks, radiators, "
    "coat hangers, hanging clothes, ceiling fixtures, objects, clutter, "
    "flat walls, uniform color, no shadows, overexposed, washed out, painted over"
)


# ── image helpers ─────────────────────────────────────────────────────────────

def _b64(path: Path) -> tuple[str, str]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return data, mime


# ── backend calls ─────────────────────────────────────────────────────────────

def _call_qwen(ref_path: Path, timeout: int = 300) -> bytes:
    from Qwen.image_edit_adapter import edit_image

    ref_b64, _ = _b64(ref_path)
    try:
        from PIL import Image as _PIL
        W, H = _PIL.open(ref_path).size
    except Exception:
        W, H = 1296, 968

    # Pass 1 — aggressive removal (high cfg_scale)
    print(f"[gen_empty] qwen pass1 (removal)  ref={ref_path.name}  {W}×{H}")
    pass1 = edit_image({
        "prompt":              PROMPT_REMOVE,
        "negative_prompt":     NEG_PROMPT,
        "reference_image":     ref_b64,
        "width":               W,
        "height":              H,
        "num_inference_steps": 35,
        "true_cfg_scale":      7.5,   # high: force complete object removal
        "num_images":          1,
    }, backend="qwen", timeout=timeout)
    pass1_b64 = pass1["images"][0]

    # Pass 2 — inpaint cleanup (low cfg_scale): blend patches, preserve textures
    print(f"[gen_empty] qwen pass2 (inpaint cleanup)  {W}×{H}")
    pass2 = edit_image({
        "prompt":              PROMPT_INPAINT,
        "negative_prompt":     NEG_PROMPT,
        "reference_image":     pass1_b64,
        "width":               W,
        "height":              H,
        "num_inference_steps": 25,
        "true_cfg_scale":      3.0,   # low: preserve existing surfaces faithfully
        "num_images":          1,
    }, backend="qwen", timeout=timeout)
    return base64.b64decode(pass2["images"][0])


# Single-image empty-room prompt. Two findings shaped this:
#   1. We do NOT send render_final — given the gray structural render, Gemini
#      reproduces it (flat gray box) instead of emptying the real photo.
#   2. The prompt is deliberately SHORT. Long, elaborate instructions make Gemini
#      disengage and return a near-copy (it keeps large dominant furniture like a
#      central sofa). A concise "erase furniture, extend floor/walls" prompt
#      engages it far better — empirically the difference between keeping vs
#      removing a stubborn sofa.
PROMPT_GEMINI_SINGLE = (
    "Erase all furniture, objects, lamps AND windows/glass doors from this room — "
    "including any wall-mounted art, mirrors, TVs, decor, AND all ceiling lights, "
    "chandeliers, pendant lamps and hanging light fixtures — and extend the real "
    "wall, floor and ceiling materials to cover where they were. Leave the walls "
    "and ceiling SOLID, plain and bare; do NOT add any new pictures, lights or "
    "decorations — fill every window, door opening and removed object with the "
    "SAME surrounding wall material. Keep the real wall colours, floor materials "
    "and trim unchanged. "
    # Structure lock — this is an EDIT of the photo, not a re-imagining. Gemini's
    # other failure mode is returning a *different* furnished room (re-styled
    # furniture, shifted walls, new perspective) which both keeps furniture and
    # breaks the geometry the backprojection relies on.
    "CRITICAL: this is an EDIT of the SAME photo — keep the room's architecture "
    "and camera EXACTLY as shown: identical wall angles and positions, identical "
    "corners, ceiling line and floor line, identical perspective, vanishing points "
    "and framing. Do NOT move, rotate, re-shape, re-scale, re-style or "
    "re-decorate the room, and do NOT invent any new furniture — only delete what "
    "is there. Output one photorealistic image of the SAME room from the SAME "
    "viewpoint, completely empty with solid bare walls and a bare ceiling."
)


# Iterative-cleanup prompt run on the OUTPUT to strip stubborn leftovers (a
# dominant sofa survives explicit "remove the sofa" prompts but vanishes with the
# "vacant apartment before move-in" framing — see feedback_iterative_empty_room_gen)
# while holding the architecture/perspective EXACTLY fixed.
PROMPT_VACANT = (
    "Show this EXACT same room as a completely BARE, VACANT shell before move-in — "
    "an empty real-estate listing photo stripped to construction state. Remove "
    "EVERYTHING that is not the room's own walls, floor and ceiling: ALL furniture, "
    "sofas, chairs, tables, beds, cabinets, lamps, rugs, plants and decorations, AND "
    "ALL windows, glass doors, wall lights / sconces, radiators, framed art and "
    "mirrors. Fill every removed region — INCLUDING each window and door opening — "
    "with the SAME surrounding bare wall (or floor) material so the walls are SOLID, "
    "plain and unbroken, with no window holes, no plants and nothing hanging. Keep "
    "the wall colour/pattern, the floor material, the baseboards, the ceiling, the "
    "camera viewpoint and the perspective EXACTLY the same. Do NOT add, move, "
    "rearrange, restyle, re-scale or repaint anything. Output a bare empty shell only."
)


def _gemini_edit_bytes(img_bytes: bytes, prompt: str, api_key: str,
                       model: str, timeout: int = 300) -> bytes | None:
    """Single Gemini image-edit from in-memory bytes (used for the iterative
    output-cleanup passes). Routed through the shared adapter so it works with
    BOTH the Google-direct key (GEMINI_API_KEY) and the NVIDIA gateway (a '/'
    model id + SCENEWEAVE_GEMINI_GATEWAY_KEY/NVIDIA_API_KEY). Returns PNG bytes."""
    from Qwen.image_edit_adapter import edit_image
    resp = edit_image({
        "prompt": prompt, "negative_prompt": "",
        "reference_image": base64.b64encode(img_bytes).decode(),
    }, backend="gemini", timeout=timeout)
    imgs = resp.get("images") or []
    return base64.b64decode(imgs[0]) if imgs else None


def _call_gemini(render_path: Path, ref_path: Path,
                 model: str, api_key: str, timeout: int = 300) -> bytes:
    """Base empty-room generation from the real photo, routed through the shared
    adapter (Google-direct OR NVIDIA gateway). Single image (the real photo)
    only — see PROMPT_GEMINI_SINGLE note above."""
    from Qwen.image_edit_adapter import edit_image
    ref_b64, _ref_mime = _b64(ref_path)
    print(f"[gen_empty] gemini/{model}  ref={ref_path.name}  (adapter, photo-only)")
    resp = edit_image({
        "prompt": PROMPT_GEMINI_SINGLE, "negative_prompt": "",
        "reference_image": ref_b64,
    }, backend="gemini", timeout=timeout)
    imgs = resp.get("images") or []
    if not imgs:
        raise RuntimeError("no image in Gemini response")
    return base64.b64decode(imgs[0])


# ── scene processing ──────────────────────────────────────────────────────────

def _emptiness_score(candidate_bytes: bytes, ref_path: Path) -> float:
    """How much the generated empty-room differs from the original photo, as the
    mean absolute grayscale difference (0–255) at 256×256.

    Gemini's failure mode is returning ~the original (furniture kept) → LOW diff.
    A real emptying changes the furniture regions to wall → HIGH diff. Empirically
    furniture-kept scenes score ≤16 and emptied scenes ≥23, so ~20 separates them
    (and it is NOT fooled by patterned/wallpapered walls, which score high)."""
    import numpy as np
    import cv2
    arr = np.frombuffer(candidate_bytes, np.uint8)
    cand = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if cand is None or ref is None:
        return 0.0
    cand = cv2.resize(cand, (256, 256)).astype(float)
    ref = cv2.resize(ref, (256, 256)).astype(float)
    return float(np.abs(cand - ref).mean())


def process_scene(scene_dir: Path, backend: str, model: str,
                  api_key: str | None, skip_existing: bool = False,
                  ref_path: "Path | str | None" = None,
                  verify_thresh: float = 20.0, max_attempts: int = 3,
                  refine_passes: int = 2) -> bool:
    render = scene_dir / "render_final.png"
    out    = scene_dir / OUTPUT_NAME

    if ref_path is not None:
        ref = Path(ref_path)
    else:
        ref = None
        for ext in (".jpeg", ".jpg", ".png", ".avif", ".webp"):
            cand = scene_dir / (scene_dir.name + ext)
            if cand.exists():
                ref = cand
                break
        if ref is None:
            for ext in (".jpeg", ".jpg", ".png", ".avif", ".webp"):
                candidates = sorted(scene_dir.glob(f"*{ext}"))
                if candidates:
                    ref = candidates[0]
                    break

    if ref is None or not ref.exists():
        print(f"[gen_empty] {scene_dir.name}: no reference image — skip")
        return False
    if not render.exists():
        print(f"[gen_empty] {scene_dir.name}: render_final.png missing — skip")
        return False
    if skip_existing and out.exists():
        print(f"[gen_empty] {scene_dir.name}: {OUTPUT_NAME} exists — skip")
        return True

    try:
        # Accept EITHER a Google-direct key (api_key) OR the NVIDIA gateway key
        # (routed via the adapter) — the gemini calls go through edit_image now.
        if backend == "gemini" and not (api_key
                or os.environ.get("SCENEWEAVE_GEMINI_GATEWAY_KEY")
                or os.environ.get("NVIDIA_API_KEY")):
            raise RuntimeError("No Gemini key (set GEMINI_API_KEY, or "
                               "SCENEWEAVE_GEMINI_GATEWAY_KEY/NVIDIA_API_KEY for gateway)")

        # Generate, verify it's actually emptied, retry up to max_attempts —
        # Gemini intermittently returns the room with furniture still in it.
        best_bytes, best_score = None, -1.0
        attempts = max(1, max_attempts) if verify_thresh > 0 else 1
        for attempt in range(1, attempts + 1):
            if backend == "gemini":
                img_bytes = _call_gemini(render, ref, model, api_key)
            else:
                img_bytes = _call_qwen(ref)
            score = _emptiness_score(img_bytes, ref) if verify_thresh > 0 else 999.0
            print(f"[gen_empty] {scene_dir.name}: attempt {attempt}/{attempts}  "
                  f"emptiness={score:.1f} (need ≥{verify_thresh:.0f})")
            if score > best_score:
                best_score, best_bytes = score, img_bytes
            if score >= verify_thresh:
                break

        out.write_bytes(best_bytes)
        print(f"[gen_empty] {scene_dir.name}: initial best emptiness={best_score:.1f}")

        # ── Iterative cleanup of stubborn leftovers (output → output) ──────────
        # The single-shot removal strips most furniture but commonly keeps a
        # dominant piece (sofa) and can re-style the room. Re-edit the OUTPUT with
        # the "vacant apartment" framing, which empties far more reliably and locks
        # the architecture/perspective. Keep going only while it gets emptier.
        if backend == "gemini" and refine_passes > 0 and best_bytes is not None:
            cur_bytes = best_bytes
            for rp in range(1, refine_passes + 1):
                try:
                    ref_bytes = _gemini_edit_bytes(cur_bytes, PROMPT_VACANT, api_key, model)
                except Exception as e:
                    print(f"[gen_empty] {scene_dir.name}: refine pass {rp} ERROR — {e}")
                    break
                if ref_bytes is None:
                    print(f"[gen_empty] {scene_dir.name}: refine pass {rp} — no image")
                    break
                rscore = _emptiness_score(ref_bytes, ref)
                print(f"[gen_empty] {scene_dir.name}: refine pass {rp}  emptiness={rscore:.1f}")
                cur_bytes = ref_bytes
                if rscore >= best_score:        # keep the emptier result
                    best_score, best_bytes = rscore, ref_bytes
                    out.write_bytes(best_bytes)

        ok = best_score >= verify_thresh
        flag = "" if ok else "  ⚠ STILL FURNISHED — below threshold"
        print(f"[gen_empty] saved → {out}  ({len(best_bytes)//1024} KB)  "
              f"emptiness={best_score:.1f}{flag}")
        return ok
    except Exception as e:
        print(f"[gen_empty] {scene_dir.name}: ERROR — {e}")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="Path to a single scene directory.")
    src.add_argument("--root",  help="Root directory; process all scene subdirs.")

    ap.add_argument("--backend", default=None,
                    choices=["qwen", "gemini"],
                    help="Image-edit backend. Default: qwen (or SCENEWEAVE_IMG_EDIT env var).")
    ap.add_argument("--model", default="gemini-2.5-flash-image",
                    help="Gemini model ID (gemini backend only).")
    ap.add_argument("--skip-existing", action="store_true",
                    help=f"Skip scenes that already have {OUTPUT_NAME}.")
    ap.add_argument("--only", nargs="+", metavar="SCENE",
                    help="When using --root, only process these scene names.")
    ap.add_argument("--verify-thresh", type=float, default=20.0,
                    help="Min mean-gray-diff vs the original photo for the room to "
                         "count as emptied; retries below this. 0 disables verify.")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Max generation attempts to reach --verify-thresh "
                         "(keeps the emptiest).")
    args = ap.parse_args()

    backend  = (args.backend or os.environ.get("SCENEWEAVE_IMG_EDIT") or "gemini").lower()
    api_key  = os.environ.get("GEMINI_API_KEY") if backend == "gemini" else None
    if backend == "gemini" and not api_key:
        sys.exit("GEMINI_API_KEY not set.  Run:  export GEMINI_API_KEY='AIza...'")

    scenes: list[Path] = []
    if args.scene:
        scenes = [Path(args.scene)]
    else:
        root = Path(args.root)
        scenes = sorted(d for d in root.iterdir() if d.is_dir())
        if args.only:
            only_set = set(args.only)
            scenes = [s for s in scenes if s.name in only_set]

    print(f"[gen_empty] {len(scenes)} scene(s)  backend={backend}")
    ok = failed = 0
    for scene in scenes:
        if process_scene(scene, backend, args.model, api_key, args.skip_existing,
                         verify_thresh=args.verify_thresh,
                         max_attempts=args.max_attempts):
            ok += 1
        else:
            failed += 1
    print(f"\n[gen_empty] done — {ok} ok, {failed} failed/skipped")


if __name__ == "__main__":
    main()