"""
object_generation.py — generate 3D models for furniture objects via
Hunyuan3D (the only 3D-generation backend in this release; the module was
in this release — Hunyuan3D is now required, not optional).

Reads segment_results.json from a furniture output folder.  For each segment,
uses its inpainted image (from the inpainted subdir) if available, otherwise
the canvas PNG (from segmented/).

On a Hunyuan3D failure that looks like the known "can't mesh this" signature
(HTTP 404 — glass/transparent objects deadlock the marching-cubes step and the
server answers 404 rather than crashing), this re-inpaints the SAME segment as
an opaque material via `inpaint_furniture.run(indices=[idx])` and retries once,
instead of just giving up on the object.

Outputs are saved to <furniture_dir>/objects/inpaint_<idx>_<type>.glb.

Usage:
    python -m object_placement.furniture.object_generation \\
        --furniture-dir outputs/20260331_031530/furniture \\
        --inpainted-subdir inpainted5 \\
        --indices 0 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ── Hunyuan3D API ─────────────────────────────────────────────────────────────
HUNYUAN_SERVER = os.environ.get("HUNYUAN_SERVER", "http://localhost:8081")
_HUNYUAN_TIMEOUT = int(os.environ.get("SCENEWEAVE_HUNYUAN_TIMEOUT", "300"))  # per-object /generate read timeout; lowered from 600 so a hung Hunyuan (glass/transparent objects deadlock the server at 0% util) fails fast instead of stalling the worker ~10min/object

_FLAT_TYPES = {"carpet", "rug"}

VLM_API_URL = "http://localhost:8080/v1/chat/completions"


class HunyuanUnmeshableError(Exception):
    """Hunyuan3D answered 404 — the known can't-mesh-this signature (typically
    a glass/transparent object deadlocking marching cubes). A caller should
    re-inpaint the object as opaque and retry, not just give up."""


# ── Hunyuan3D generation ──────────────────────────────────────────────────────

def _hunyuan_available(server: str = HUNYUAN_SERVER) -> bool:
    """Return True if the Hunyuan3D API server is reachable."""
    try:
        requests.get(f"{server}/docs", timeout=3)
        return True
    except Exception:
        return False


def _generate_hunyuan(image_path: Path, glb_path: Path,
                      server: str = HUNYUAN_SERVER,
                      texture: bool = True) -> bool:
    """Generate a GLB via the Hunyuan3D API server. Returns True on success.

    Under SCENEWEAVE_HUNYUAN_ONLY, if the request fails because
    the server is down (it OOM-restarts under the tight RAM cgroup), wait for a
    watchdog to bring it back and retry, rather than giving up (which would leave
    the object un-generated)."""
    import base64
    import time as _time

    hunyuan_only = bool(os.environ.get("SCENEWEAVE_HUNYUAN_ONLY"))
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    attempts = 8 if hunyuan_only else 1
    for attempt in range(1, attempts + 1):
        print(f"  [hunyuan] Generating via {server} …"
              + (f" (try {attempt}/{attempts})" if attempts > 1 else ""))
        try:
            # Octree resolution controls the marching-cubes density grid: higher =
            # finer, more continuous surfaces (fewer shattered fragments) at the
            # cost of GPU/time.  Default 128; raise via SCENEWEAVE_HUNYUAN_OCTREE
            # (e.g. 384) when a large GPU is available.
            _octree = int(os.environ.get("SCENEWEAVE_HUNYUAN_OCTREE", "128") or 128)
            _req = {"image": img_b64, "texture": texture, "octree_resolution": _octree}
            _fc = os.environ.get("SCENEWEAVE_HUNYUAN_FACECOUNT")
            if _fc:
                _req["face_count"] = int(_fc)
            # Shape-diffusion steps: the server defaults to a fast preview value
            # (5).  Raise via SCENEWEAVE_HUNYUAN_STEPS (e.g. 25) for markedly
            # better shape fidelity; SCENEWEAVE_HUNYUAN_GUIDANCE tunes guidance.
            _steps = os.environ.get("SCENEWEAVE_HUNYUAN_STEPS")
            if _steps:
                _req["num_inference_steps"] = int(_steps)
            _guid = os.environ.get("SCENEWEAVE_HUNYUAN_GUIDANCE")
            if _guid:
                _req["guidance_scale"] = float(_guid)
            resp = requests.post(
                f"{server}/generate",
                json=_req,
                timeout=_HUNYUAN_TIMEOUT,
            )
            if resp.status_code == 404:
                # Known can't-mesh-this signature (glass/transparent objects
                # deadlock marching cubes; the server answers 404 for THIS
                # image rather than being down) — not a "wait and retry"
                # situation. Let the caller decide (re-inpaint + retry).
                raise HunyuanUnmeshableError(
                    f"Hunyuan3D returned 404 for {image_path.name} — "
                    "likely an unmeshable (glass/transparent) object"
                )
            resp.raise_for_status()
            glb_path.write_bytes(resp.content)
            print(f"  [hunyuan] Saved → {glb_path}  ({len(resp.content) // 1024} KB)")
            return True
        except HunyuanUnmeshableError:
            raise
        except Exception as e:
            print(f"  [hunyuan] FAILED: {e}")
            if not hunyuan_only or attempt == attempts:
                return False
            # Wait for the watchdog to restart the server, then retry.
            print("  [hunyuan] HUNYUAN_ONLY: waiting for server to come back…")
            for _ in range(40):  # up to ~10 min
                _time.sleep(15)
                if _hunyuan_available(server):
                    break
    return False


# ── VLM quality check ────────────────────────────────────────────────────────

def _render_view(verts: np.ndarray, vc: np.ndarray,
                 el_deg: float, az_deg: float, size: int) -> np.ndarray:
    """Render a single orthographic view. Returns (size, size, 3) uint8 array."""
    centre = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts_c = verts - centre

    el, az = np.radians(el_deg), np.radians(az_deg)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    R = Rx @ Ry
    verts_r = verts_c @ R.T

    span = max(np.ptp(verts_r[:, 0]), np.ptp(verts_r[:, 1])) * 1.15
    if span < 1e-6:
        return np.full((size, size, 3), 240, dtype=np.uint8)
    sc = (size - 20) / span
    mid_x = (verts_r[:, 0].min() + verts_r[:, 0].max()) / 2.0
    mid_y = (verts_r[:, 1].min() + verts_r[:, 1].max()) / 2.0

    px = ((verts_r[:, 0] - mid_x) * sc + size / 2.0).astype(np.int32)
    py = (-(verts_r[:, 1] - mid_y) * sc + size / 2.0).astype(np.int32)
    pz = verts_r[:, 2]

    order = np.argsort(pz)
    buf = np.full((size, size, 3), 240, dtype=np.uint8)

    valid = (px >= 1) & (px < size - 1) & (py >= 1) & (py < size - 1)
    for i in order:
        if not valid[i]:
            continue
        x, y = int(px[i]), int(py[i])
        buf[y-1:y+1, x-1:x+1] = vc[i]

    return buf


def _render_glb_preview(glb_path: Path, size: int = 384) -> Image.Image | None:
    """Render a GLB mesh from multiple views for VLM inspection.

    Produces a 2-up image: 3/4 elevated view (left) + side view (right).
    The side view makes ground-plane slabs clearly visible.
    Returns an RGB PIL Image or None on failure.
    """
    try:
        import trimesh

        mesh = trimesh.load(str(glb_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        verts = mesh.vertices.astype(np.float64)

        vc = None
        try:
            if hasattr(mesh.visual, "to_color"):
                vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.uint8)
        except Exception:
            pass
        if vc is None or len(vc) != len(verts):
            vc = np.full((len(verts), 3), 160, dtype=np.uint8)

        # View 1: 3/4 elevated  |  View 2: pure side (0° elevation, 0° azimuth)
        v1 = _render_view(verts, vc, el_deg=25, az_deg=30, size=size)
        v2 = _render_view(verts, vc, el_deg=0,  az_deg=0,  size=size)

        # Stitch side by side
        combined = np.concatenate([v1, v2], axis=1)
        return Image.fromarray(combined)

    except Exception as e:
        print(f"  [vlm_check] render failed: {e}")
        return None


def _encode_pil(img: Image.Image, max_side: int = 512) -> str:
    """Downscale + base64-encode a PIL Image as PNG."""
    import base64
    from io import BytesIO

    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _vlm_mesh_quality_check(
    glb_path: Path,
    source_img_path: Path,
    obj_type: str,
) -> dict:
    """Ask VLM whether the generated 3D mesh looks correct.

    Checks:
      1. Does the mesh have an unwanted ground plane / floor slab attached?
      2. Does the mesh look like the input object (shape, proportions)?

    Returns {"ok": bool, "has_ground": bool, "looks_like_input": bool,
             "reason": str}.
    Defaults to {"ok": True} if VLM is unavailable.
    """
    import re

    # Render the mesh and save preview for debugging
    preview = _render_glb_preview(glb_path)
    if preview is None:
        return {"ok": True, "reason": "render failed — skipping check"}

    preview_path = glb_path.with_suffix(".vlm_preview.png")
    preview.save(str(preview_path))
    print(f"  [vlm_check] preview saved → {preview_path}")

    source_img = Image.open(source_img_path).convert("RGB")

    prompt = f"""You are inspecting a 3D mesh reconstruction of a {obj_type.replace('_', ' ')}.

Image 1: the original 2D reference image showing ONLY the {obj_type.replace('_', ' ')} on a grey background.
Image 2: two views of the reconstructed 3D mesh — left is a 3/4 elevated view, right is a SIDE view.

Check for a **ground plane artifact**: a large flat horizontal slab or disc
ATTACHED to the bottom of the mesh that extends WELL BEYOND the footprint of
the object itself. This is a common reconstruction artifact where the floor
surface gets baked into the object mesh.

IMPORTANT — these are NOT ground planes (do NOT flag these):
- The object's own flat bottom or base
- Furniture feet/legs resting on the same horizontal level
- A thin bottom edge visible in side view — legs of a table naturally
  end at the same Y level, this is normal
- Any part that is part of the actual object design

A ground plane IS: a solid filled slab/platform that is clearly separate
from the object, extends outward like a floor, and would not be part of
the real furniture piece.

When in doubt, answer has_ground: false. Only flag obvious large slabs.

Also check if the shape roughly matches the reference.

Respond with ONLY a JSON object, no markdown:
{{"has_ground": true/false, "looks_like_input": true/false, "reason": "brief explanation"}}"""

    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_pil(source_img)}"}})
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_pil(preview)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        resp = requests.post(VLM_API_URL, json=payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            has_ground = bool(result.get("has_ground", False))
            looks_like = bool(result.get("looks_like_input", True))
            reason = result.get("reason", "")
            ok = not has_ground and looks_like
            print(f"  [vlm_check] ground={has_ground}  shape_match={looks_like}  "
                  f"ok={ok}  reason={reason}")
            return {"ok": ok, "has_ground": has_ground,
                    "looks_like_input": looks_like, "reason": reason}
    except Exception as e:
        print(f"  [vlm_check] VLM call failed: {e}")

    return {"ok": True, "reason": "VLM unavailable — accepting"}


# ── Main pipeline ──────────────────────────────────────────────────────────────

_MAX_REGEN_ATTEMPTS = 3   # max times to regenerate a rejected mesh


def _vlm_ok(glb_path: Path, src: Path, obj_type: str,
            vlm_check: bool) -> bool:
    """Run VLM quality check if enabled. Returns True if mesh passes."""
    if not vlm_check or obj_type in _FLAT_TYPES:
        return True
    qc = _vlm_mesh_quality_check(glb_path, src, obj_type)
    return qc.get("ok", True)


def _reinpaint_opaque(output_dir: Path, idx: int, obj_type: str) -> Path | None:
    """Re-inpaint one segment as an opaque material and return its new image
    path, or None on failure. Used when Hunyuan3D 404s on a segment (the
    known glass/transparent-object signature)."""
    from object_placement.furniture.inpaint_furniture import run as run_inpaint

    f_dir     = output_dir / "furniture"
    results_p = f_dir / "segment_results.json"
    with open(results_p) as f:
        data = json.load(f)
    for s in data["segments"]:
        if s["index"] == idx:
            s["inpaint_prompt_override"] = (
                f"A complete, well-lit product photo of a single {obj_type}, "
                "made of solid opaque material (wood, fabric, or matte painted "
                "metal — NOT glass, NOT transparent, NOT translucent), "
                "centered on a plain neutral background, matching the object's "
                "original color and style."
            )
            break
    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [reinpaint] regenerating idx={idx} as opaque material …")
    run_inpaint(output_dir, indices=[idx], force=True)

    with open(results_p) as f:
        data = json.load(f)
    for s in data["segments"]:
        if s["index"] == idx and s.get("inpaint_file"):
            return f_dir / "inpainted" / s["inpaint_file"]
    return None


def run(
    furniture_dir: str | Path,
    inpainted_subdir: str = "inpainted",
    indices: list[int] | None = None,
    vlm_check: bool = False,
) -> Path:
    f_dir       = Path(furniture_dir)
    results_p   = f_dir / "segment_results.json"
    inpaint_dir = f_dir / inpainted_subdir
    seg_dir     = f_dir / "segmented"
    out_dir     = f_dir / "objects"

    if not results_p.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_furniture first.\n"
            f"Expected: {results_p}"
        )

    with open(results_p) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[object_gen] No segments — nothing to do.")
        return out_dir

    if indices is not None:
        segments_to_run = [s for s in segments if s["index"] in indices]
    else:
        segments_to_run = segments

    out_dir.mkdir(exist_ok=True)

    # Hunyuan3D is required; there is no second backend. Wait for the server
    # (a watchdog restarts it after its periodic OOM under the RAM cgroup) rather
    # than failing the whole run outright.
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
                f"Hunyuan3D server ({HUNYUAN_SERVER}) never came up — "
                "no fallback backend is configured in this release."
            )

    print(f"[object_gen] Processing {len(segments_to_run)} segment(s)."
          + (" (VLM check enabled)" if vlm_check else ""))
    print(f"[object_gen] Generator     : Hunyuan3D")
    print(f"[object_gen] Inpainted dir : {inpaint_dir}")

    success = 0
    for seg in segments_to_run:
        idx      = seg["index"]
        obj_type = seg.get("type", "other")
        stem     = f"inpaint_{idx:02d}_{obj_type}"
        glb_path = out_dir / f"{stem}.glb"

        print(f"\n[object_gen] {idx:02d} {obj_type}")

        # Prefer inpainted image; fall back to canvas
        src: Path | None = None
        if seg.get("inpaint_file"):
            candidate = inpaint_dir / seg["inpaint_file"]
            if candidate.exists():
                src = candidate
                print(f"  using inpainted: {candidate}")
        if src is None and seg.get("canvas_file"):
            candidate = seg_dir / seg["canvas_file"]
            if candidate.exists():
                src = candidate
                print(f"  inpainted not found — using canvas: {candidate.name}")
        if src is None:
            print("  no source image found — skipping.")
            continue

        # ── Check existing GLB ───────────────────────────────────────────────
        if glb_path.exists():
            if vlm_check and obj_type not in _FLAT_TYPES:
                print(f"  checking existing {glb_path.name} …")
                if _vlm_ok(glb_path, src, obj_type, vlm_check):
                    print(f"  [vlm] OK — keeping {glb_path.name}")
                    seg["glb_file"] = str(glb_path.relative_to(f_dir))
                    success += 1
                    continue
                else:
                    print(f"  [vlm] REJECTED existing — will regenerate with Hunyuan3D")
                    rejected_path = glb_path.with_suffix(".rejected.glb")
                    glb_path.rename(rejected_path)
            else:
                print(f"  reusing existing {glb_path.name}")
                seg["glb_file"] = str(glb_path.relative_to(f_dir))
                success += 1
                continue

        # ── Flat-type early skip ─────────────────────────────────────────────
        # Carpets / rugs are always rendered as a flat quad in place_furniture
        # (_render_flat_carpet_quad), never from a generated GLB.
        if obj_type in _FLAT_TYPES:
            print(f"  flat type — handled by runtime quad renderer, "
                  f"no GLB needed (skipping)")
            continue

        # ── Generate ─────────────────────────────────────────────────────────
        # Try Hunyuan3D; on the known unmeshable-object 404, re-inpaint this
        # segment as opaque and retry once before falling through to a normal
        # VLM-rejection regen loop.
        max_attempts = _MAX_REGEN_ATTEMPTS if vlm_check else 1
        generated = False
        reinpainted_for_404 = False

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"  [regen] attempt {attempt}/{max_attempts} (Hunyuan3D)")

            gen_ok = False
            try:
                gen_ok = _generate_hunyuan(src, glb_path)
            except HunyuanUnmeshableError as e:
                print(f"  [hunyuan] {e}")
                if reinpainted_for_404:
                    print("  [reinpaint] already retried once — giving up on this object")
                    break
                reinpainted_for_404 = True
                new_src = _reinpaint_opaque(f_dir.parent, idx, obj_type)
                if new_src is None:
                    print("  [reinpaint] failed — giving up on this object")
                    break
                src = new_src
                try:
                    gen_ok = _generate_hunyuan(src, glb_path)
                except HunyuanUnmeshableError as e2:
                    print(f"  [hunyuan] still unmeshable after re-inpaint: {e2}")
                    gen_ok = False

            if not gen_ok:
                print(f"  FAILED — Hunyuan3D could not generate this object")
                break

            print(f"  saved → {glb_path}")

            # VLM quality check
            if _vlm_ok(glb_path, src, obj_type, vlm_check):
                generated = True
                break
            else:
                print(f"  [vlm] REJECTED (attempt {attempt}) — will retry with Hunyuan3D")
                rejected_path = out_dir / f"{stem}.rejected_v{attempt}.glb"
                glb_path.rename(rejected_path)

        if generated:
            for s in data["segments"]:
                if s["index"] == idx:
                    s["glb_file"] = str(glb_path.relative_to(f_dir))
                    break
            success += 1
        elif vlm_check:
            print(f"  [check] all {max_attempts} attempts rejected — skipping idx={idx}")

    # Persist glb_file paths back into segment_results.json
    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[object_gen] Done. {success}/{len(segments_to_run)} models → {out_dir}/")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 3D GLB models for furniture objects via Hunyuan3D."
    )
    ap.add_argument(
        "--furniture-dir", required=True,
        help="Path to the furniture output folder (contains segment_results.json)",
    )
    ap.add_argument(
        "--inpainted-subdir", default="inpainted",
        help="Subdirectory under furniture-dir containing inpainted PNGs (default: inpainted)",
    )
    ap.add_argument(
        "--indices", nargs="+", type=int, default=None,
        help="Which segment indices to process (default: all). E.g. --indices 0 1",
    )
    ap.add_argument(
        "--vlm-check", action="store_true",
        help="Run VLM quality check on generated (and existing) meshes. "
             "Rejects meshes with ground planes or wrong shape and regenerates.",
    )
    args = ap.parse_args()
    run(
        furniture_dir=args.furniture_dir,
        inpainted_subdir=args.inpainted_subdir,
        indices=args.indices,
        vlm_check=args.vlm_check,
    )


if __name__ == "__main__":
    main()
