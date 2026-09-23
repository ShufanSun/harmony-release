"""
texture_refine.py — Generate texture maps from a reference room photo.

Step 1: Qwen VLM reads the reference image and outputs a texture description
        for each visible surface (floor, walls, ceiling).
Step 2: Take the floor description + reference image → Qwen Image Edit
        → seamless texture map matching the real surface.

Usage
-----
    python -m Qwen.texture_refine data/indoor_images/office4.png [--out outputs/textures]
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import requests
from PIL import Image

VLM_URL = "http://localhost:8080/v1/chat/completions"
QWEN_IMAGE_EDIT_MODEL = "Qwen/Qwen-Image-Edit"


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime(path: str | Path) -> str:
    return "image/png" if str(path).lower().endswith(".png") else "image/jpeg"


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


# ── Step 1: VLM reads reference image → texture descriptions ─────────────────

def describe_surfaces(ref_image_path: str | Path) -> dict[str, str]:
    """
    Send the reference room photo to the Qwen VLM.
    Returns a dict: surface -> description string.
    Surfaces: floor, ceiling, back_wall, left_wall, right_wall
    (only the ones visible in the image).
    """
    ref_b64 = _encode(ref_image_path)
    mime    = _mime(ref_image_path)

    prompt = """\
You are given a photo of a real room interior.

TASK: Identify every distinct surface visible in the image and describe each one's
texture precisely enough for a diffusion model to reproduce it as a seamless tile.

IMPORTANT: Two walls that look DIFFERENT (different colour, material, or pattern)
must be listed separately under their own key (e.g. back_wall vs left_wall).
Do NOT describe all walls with the same text if they visually differ.

For EACH visible surface write one line in this exact format:
  <surface>: <description>

Where <surface> is one of: floor, ceiling, back_wall, left_wall, right_wall
And <description> covers:
  - material (e.g. oak parquet, ceramic tile, painted plaster, concrete)
  - colour   (be specific: e.g. warm light beige, dark charcoal grey, pale sage green)
  - pattern  (e.g. herringbone, straight planks, uniform, grid)
  - finish   (matte, semi-gloss, polished, rough)
  - rough scale (e.g. planks ~8 cm wide, tiles ~30 cm)

Only include surfaces that are actually visible. Output nothing else.
"""

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{ref_b64}"}},
        ]}],
        "max_tokens": 400,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    resp = requests.post(VLM_URL, json=payload, timeout=90)
    resp.raise_for_status()
    raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])

    surfaces: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, desc = line.partition(":")
        key  = key.strip().lower().replace(" ", "_")
        desc = desc.strip()
        if key and desc:
            surfaces[key] = desc

    return surfaces


# ── Step 2: Qwen Image Edit → seamless texture map ───────────────────────────

def _build_edit_prompt(surface: str, description: str) -> str:
    if surface in ("floor", "ceiling"):
        view_hint = (
            "perfectly flat top-down orthographic view, "
            "all planks/tiles/lines must be exactly horizontal or vertical, "
            "zero perspective, zero foreshortening"
        )
    else:
        view_hint = (
            "perfectly flat front-on orthographic view, "
            "all horizontal features (grout lines, brick rows, wood grain) "
            "must be exactly horizontal — zero diagonal lines, "
            "zero perspective, zero foreshortening"
        )
    return (
        f"Extract the {surface.replace('_', ' ')} texture from this room photo and convert it "
        f"into a seamless tileable texture map. "
        f"Material: {description}. "
        f"Requirements: {view_hint}. "
        f"Uniform flat lighting, no shadows, no furniture, no depth cues, "
        f"no perspective distortion, pure repeating surface pattern only, square image."
    )


def generate_all_textures(
    ref_image_path: str | Path,
    surfaces: dict[str, str],       # surface -> VLM description
    out_dir: str | Path,
    stem: str,                       # filename prefix, e.g. "office4"
    size: int = 1024,
) -> dict[str, Path]:
    """
    Load QwenImageEditPipeline once, then generate a texture map for every
    surface in `surfaces`.  Returns a dict of surface -> saved path.
    """
    import torch
    from diffusers import QwenImageEditPipeline

    print("[texture_refine] Loading QwenImageEditPipeline...")
    pipeline = QwenImageEditPipeline.from_pretrained(
        QWEN_IMAGE_EDIT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )

    ref_image = Image.open(ref_image_path).convert("RGB").resize((size, size))
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for surface, description in surfaces.items():
        edit_prompt = _build_edit_prompt(surface, description)
        print(f"\n[texture_refine] --- {surface.upper()} ---")
        print(f"[texture_refine] Prompt: {edit_prompt}")

        inputs = {
            "image":               ref_image,
            "prompt":              edit_prompt,
            "negative_prompt":     "shadows, furniture, perspective, depth, blurry, dark corners",
            "num_inference_steps": 50,
            "true_cfg_scale":      4.0,
            "generator":           torch.manual_seed(42),
        }

        with torch.inference_mode():
            output = pipeline(**inputs)

        out_path = out_dir / f"{stem}_{surface}_texture.png"
        output.images[0].resize((size, size)).save(out_path)
        print(f"[texture_refine] Saved → {out_path}")
        results[surface] = out_path

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_image", help="Reference room photo")
    parser.add_argument("--out",     default="outputs/textures",
                        help="Output directory (default: outputs/textures)")
    parser.add_argument("--size",    type=int, default=1024,
                        help="Texture size in pixels (default 1024)")
    args = parser.parse_args()

    ref     = Path(args.ref_image)
    out_dir = Path(args.out)

    # Step 1: VLM reads reference image → description per surface
    print("[texture_refine] Step 1: VLM reading reference image...")
    surfaces = describe_surfaces(ref)
    if not surfaces:
        print("[texture_refine] VLM returned no surface descriptions. Exiting.")
        sys.exit(1)

    print("[texture_refine] Detected surfaces:")
    for s, d in surfaces.items():
        print(f"  {s}: {d}")

    # Step 2: generate texture maps for all detected surfaces
    print("\n[texture_refine] Step 2: Generating texture maps...")
    results = generate_all_textures(
        ref_image_path=ref,
        surfaces=surfaces,
        out_dir=out_dir,
        stem=ref.stem,
        size=args.size,
    )

    print(f"\n[texture_refine] Done. {len(results)} texture(s) saved to {out_dir}/")
