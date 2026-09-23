"""
inpaint_ceiling_objects.py — complete each segmented ceiling fixture into a
clean, isolated product image suitable for 3D reconstruction.

Counterpart to ``wall_mounted/inpaint_wall_objects_keep_glass.py``.  Each
``ceiling/segmented/segment_NN_<type>_canvas.png`` (object on a grey square
canvas) is run through the image-edit backend with a ceiling-fixture prompt that
asks for the COMPLETE fixture — including the cord / chain / rod and the ceiling
canopy at the top — on a plain neutral background, so the downstream Hunyuan3D /
the 3D generator step gets the whole hanging shape rather than a cropped lamp body.

Reuses the low-level edit call (``_raw_edit``) and IO helpers from the
wall-mounted inpainter, so it honours the same SCENEWEAVE_IMG_EDIT backend
selection (Qwen image-edit / Gemini).

Outputs written to <output_dir>/ceiling/inpainted/.

Usage:
    python -m object_placement.ceiling.inpaint_ceiling_objects \\
        --output-dir outputs/front3d/rgb_003200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from object_placement.wall_mounted.inpaint_wall_objects_keep_glass import (
    _raw_edit,
    _GLOBAL_NEG,
    _DEFAULT_CFG,
)

# ── Per-type completion prompt ────────────────────────────────────────────────
_TYPE_DESCRIPTOR: dict[str, str] = {
    "pendant_light": "a single hanging pendant light fixture, including its full "
                     "suspension cord/rod and the small ceiling canopy at the top",
    "chandelier":    "a complete chandelier with all of its arms, bulbs and the "
                     "chain/rod and ceiling canopy at the top",
    "ceiling_lamp":  "a flush-mount ceiling light fixture seen from slightly below, "
                     "complete with its mounting plate",
    "ceiling_fan":   "a complete ceiling fan with all blades, the motor housing and "
                     "the down-rod and ceiling mount at the top",
    "track_light":   "the ceiling spotlight fixture EXACTLY as in the reference — if "
                     "the reference shows only a single small recessed or flush "
                     "downlight, render just that one flat recessed light (NO rail, "
                     "NO extra spotlight heads, NO rod); include a rail with multiple "
                     "heads ONLY if the reference clearly shows them",
    "other":         "a complete ceiling light fixture including its suspension and "
                     "ceiling mount at the top",
}


# When the reference shows the fixture mounted close to the ceiling with NO visible
# suspension (flush drum chandeliers, semi-flush fixtures — 007406), forcing a long
# cord/rod into the inpaint produces a 3D object whose thin stem is hard to align on
# the ceiling plane. Use a rod-free descriptor in that case.
_NOROD_DESCRIPTOR: dict[str, str] = {
    "pendant_light": "a single pendant / drum light fixture mounted close to the ceiling",
    "chandelier":    "a complete chandelier with all of its arms and bulbs, mounted "
                     "close to the ceiling",
    "other":         "a complete ceiling light fixture mounted close to the ceiling",
}


def _has_visible_rod(mask_path: Path) -> bool:
    """True if the segmented fixture has a thin suspension stem (cord / rod / chain)
    descending from the ceiling — a narrow vertical extension above the main body.
    False for flush / drum fixtures shown mounted close to the ceiling with no
    visible rod, where fabricating one yields a hard-to-align 3D object (007406)."""
    try:
        import numpy as np
        m = np.array(Image.open(mask_path).convert("L")) > 127
    except Exception:
        return True   # default to the old behaviour (keep the rod)
    if m.sum() < 50:
        return True
    rows = np.where(m.any(axis=1))[0]
    if len(rows) < 8:
        return True
    y0, y1 = int(rows[0]), int(rows[-1])
    H = y1 - y0 + 1
    widths = m.sum(axis=1).astype(float)
    body_w = float(widths[y0:y1 + 1].max())
    if body_w <= 0:
        return True
    # A real rod is a run of THIN rows (< 25% of body width) in the top ~45% of the
    # fixture, before the body widens out. A wide drum/flush fixture has none.
    n_top = max(1, int(0.45 * H))
    thin = int(sum(1 for y in range(y0, y0 + n_top) if 0 < widths[y] < 0.25 * body_w))
    return thin >= max(4, int(0.12 * H))


def _build_prompt(obj_type: str, phrase: str, has_rod: bool = True) -> str:
    import os as _os
    if _os.environ.get("SCENEWEAVE_INPAINT_SHORT"):
        t = obj_type.replace("_", " ")
        rod = ("Keep its full drop/rod/cord/chain above the fixture. " if has_rod
               else "Render only the compact fixture body mounted close to the ceiling — "
                    "no long cord/rod/chain above it. ")
        return (f"Extract the {t} shown and amodally complete it into a single, whole "
                f"{t} on a plain grey background. Keep its exact shape, the same number "
                f"of lamps/arms/blades, colour and materials — do not redesign it. {rod}"
                f"Remove the room, walls and any other object; nothing cropped or floating.")
    if not has_rod:
        descriptor = _NOROD_DESCRIPTOR.get(obj_type, _NOROD_DESCRIPTOR["other"])
        return (
            f"A studio product photograph of {descriptor}. "
            f"Reproduce the fixture shown in the reference EXACTLY — same shape, same "
            f"number of lamps/arms/bulbs, same materials and colour. Render ONLY the "
            f"fixture body mounted close to the ceiling; do NOT add any long cord, rod, "
            f"chain or wire above it — just the fixture and its compact ceiling mount. "
            f"Centre it on a plain, uniform neutral light-grey background. "
            f"No ceiling, no walls, no room, no floor, sharp focus, even lighting."
        )
    descriptor = _TYPE_DESCRIPTOR.get(obj_type, _TYPE_DESCRIPTOR["other"])
    return (
        f"A studio product photograph of {descriptor}. "
        f"Reproduce the fixture shown in the reference EXACTLY — same shape, same "
        f"number of lamps/arms/blades, same materials and colour. Render the COMPLETE "
        f"object, with the cord/chain/rod and ceiling mount at the very top of the "
        f"frame. Centre it on a plain, uniform neutral light-grey background. "
        f"No ceiling, no walls, no room, no floor, sharp focus, even lighting."
    )


_CEIL_NEG = (_GLOBAL_NEG + ", ceiling, wall, room, floor, furniture, window, "
             "multiple objects, cropped, cut off, frame, border")
_CEIL_NEG_NOROD = _CEIL_NEG + ", long cord, long rod, long chain, hanging wire, suspension cord"


def run(output_dir: str | Path,
        types: list[str] | None = None,
        indices: list[int] | None = None) -> Path:
    out_dir     = Path(output_dir) / "ceiling"
    seg_dir     = out_dir / "segmented"
    inpaint_dir = out_dir / "inpainted"
    results_p   = out_dir / "segment_results.json"

    if not results_p.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_ceiling_objects first.\n"
            f"Expected: {results_p}"
        )

    inpaint_dir.mkdir(parents=True, exist_ok=True)

    with open(results_p) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[ceil_inpaint] No segments — nothing to do.")
        return inpaint_dir

    for seg in segments:
        idx         = seg["index"]
        obj_type    = seg.get("type", "other")
        phrase      = seg.get("phrase", obj_type)
        canvas_file = seg.get("canvas_file")

        if types is not None and obj_type not in types:
            continue
        if indices is not None and idx not in indices:
            continue

        print(f"\n[ceil_inpaint] {idx:02d} {obj_type} — '{phrase}'")
        if not canvas_file:
            print("  no canvas_file — skipping.")
            continue
        canvas_path = seg_dir / canvas_file
        if not canvas_path.exists():
            print(f"  canvas not found: {canvas_path} — skipping.")
            continue

        out_name = f"inpaint_{idx:02d}_{obj_type}.png"
        out_path = inpaint_dir / out_name
        if out_path.exists():
            print(f"  reusing existing {out_name}")
            seg["inpaint_file"] = out_name
            continue

        canvas = Image.open(canvas_path).convert("RGB")
        # Suppress a fabricated suspension rod when the reference shows the fixture
        # mounted close to the ceiling (flush / drum) with no visible cord/rod.
        has_rod = True
        if obj_type in ("pendant_light", "chandelier", "other"):
            mask_file = seg.get("mask_file")
            if mask_file and (seg_dir / mask_file).exists():
                has_rod = _has_visible_rod(seg_dir / mask_file)
                if not has_rod:
                    print("  no visible rod in reference → suppressing fabricated cord/rod")
        prompt = _build_prompt(obj_type, phrase, has_rod=has_rod)
        neg = _CEIL_NEG if has_rod else _CEIL_NEG_NOROD
        print(f"  prompt: {prompt[:110]}")
        result = _raw_edit(canvas, prompt, neg, _DEFAULT_CFG)
        if result is None:
            print("  edit failed — skipping.")
            continue

        result.save(out_path)
        seg["inpaint_file"] = out_name
        print(f"  saved → {out_path}")

    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    done = sum(1 for s in segments if "inpaint_file" in s)
    print(f"\n[ceil_inpaint] Done. {done}/{len(segments)} fixtures inpainted → {inpaint_dir}/")
    return inpaint_dir


def main():
    ap = argparse.ArgumentParser(description="Inpaint segmented ceiling fixtures.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--indices", nargs="+", type=int, default=None)
    args = ap.parse_args()
    run(output_dir=args.output_dir, types=args.types, indices=args.indices)


if __name__ == "__main__":
    main()
