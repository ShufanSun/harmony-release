"""
segmentation_la_sam2.py — LocateAnything→SAM2 segmentation front-end for the
HARMONY pipeline (drop-in replacement for segment_wall_objects /
segment_furniture).

Because LocateAnything (env `eagle-embodied`, transformers 4.57) and SAM2
(env `lang-sam`, transformers 5.x) have conflicting deps, this runs each in its
own conda env via that env's python interpreter and hands off through a temp
boxes.json. The SAM2 stage writes the HARMONY `segment_results.json` contract
(segment_NN_<type>_{mask,canvas[,crop]}.png) into {output_dir}/{wall_mounted|
furniture}/, so the inpaint / object-gen / placement stages run unchanged.

Env overrides:
    SCENEWEAVE_LA_ENV     conda env with LocateAnything   (default eagle-embodied)
    SCENEWEAVE_SAM2_ENV   conda env with SAM2 / lang-sam  (default lang-sam)
    SCENEWEAVE_LA_DIR     dir of la_emit_boxes.py     (default /path/to/eagle/Embodied)
    SCENEWEAVE_SAM2_DIR   dir of sam2_from_boxes.py   (default /path/to/lang-segment-anything)
    SCENEWEAVE_SAM2_TYPE  sam2 checkpoint             (default sam2.1_hiera_small)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_CONDA_ROOT = os.environ.get("CONDA_ROOT", str(Path.home() / "miniconda3"))
_LA_ENV = os.environ.get("SCENEWEAVE_LA_ENV", "eagle-embodied")
_SAM2_ENV = os.environ.get("SCENEWEAVE_SAM2_ENV", "lang-sam")
_LA_DIR = os.environ.get("SCENEWEAVE_LA_DIR", "/path/to/eagle/Embodied")
_SAM2_DIR = os.environ.get("SCENEWEAVE_SAM2_DIR", "/path/to/lang-segment-anything")
_SAM2_TYPE = os.environ.get("SCENEWEAVE_SAM2_TYPE", "sam2.1_hiera_small")


def _env_python(env: str) -> str:
    p = Path(_CONDA_ROOT) / "envs" / env / "bin" / "python"
    if not p.exists():
        raise FileNotFoundError(f"conda env python not found: {p}")
    return str(p)


def _run(cmd: list[str], cwd: str) -> None:
    print(f"[la-sam2] $ (cwd={cwd})\n           {' '.join(cmd)}")
    env = os.environ.copy()
    # Reduce CUDA fragmentation for the model subprocesses (shared GPU).
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def run(output_dir: str | Path, image_path: str | Path, domain: str,
        no_verify: bool = False) -> Path:
    """Segment one domain ('wall', 'furniture' or 'decoration') via LocateAnything→SAM2.

    Returns the {output_dir}/{wall_mounted|furniture|decorations} directory
    holding the contract (segment_results.json + segmented/). For 'decoration',
    {output_dir}/decorations/decoration_analysis.json must already exist (the
    decoration→furniture map + missing-decoration names drive detection).
    """
    if domain not in ("wall", "furniture", "decoration"):
        raise ValueError(f"domain must be wall/furniture/decoration, got {domain!r}")
    # Subprocesses run with cwd in the LA / SAM2 dirs, so ALL paths must be
    # absolute (relative paths would resolve against the wrong cwd).
    output_dir = Path(output_dir).resolve()
    image_path = Path(image_path).resolve()
    dom_sub = {"wall": "wall_mounted", "furniture": "furniture",
               "decoration": "decorations"}[domain]
    dom_dir = output_dir / dom_sub
    dom_dir.mkdir(parents=True, exist_ok=True)
    boxes_json = dom_dir / "_la_boxes.json"

    # Stage 1 — LocateAnything boxes (+ VLM verify) in the LA env
    cmd1 = [_env_python(_LA_ENV), "la_emit_boxes.py",
            "--domain", domain, "--image", str(image_path), "--out", str(boxes_json)]
    if domain == "decoration":
        analysis = dom_dir / "decoration_analysis.json"
        if not analysis.exists():
            raise FileNotFoundError(f"decoration_analysis.json missing: {analysis} "
                                    "(run detect_missing_decorations first)")
        cmd1 += ["--analysis", str(analysis)]
    if no_verify or os.environ.get("SCENEWEAVE_NO_SEGMENT_VERIFY") == "1":
        cmd1.append("--no-verify")
    _run(cmd1, cwd=_LA_DIR)

    # Stage 2 — SAM2 masks → HARMONY contract in the SAM2 env
    cmd2 = [_env_python(_SAM2_ENV), "sam2_from_boxes.py",
            "--boxes", str(boxes_json), "--sam-type", _SAM2_TYPE,
            "--contract", "--output-dir", str(output_dir), "--domain", domain]
    _run(cmd2, cwd=_SAM2_DIR)

    print(f"[la-sam2] {domain} segmentation → {dom_dir}/segment_results.json")
    return dom_dir


def run_sam2_only(output_dir: str | Path, domain: str) -> Path:
    """Re-run ONLY stage 2 (SAM2 → contract) over the domain's existing
    ``_la_boxes.json``.

    Used after a recovery pass has added boxes the detector missed: the boxes
    file is the interface, so whoever produced the rectangles is irrelevant and
    masks/crops/segment_results.json are rebuilt to include them.
    """
    output_dir = Path(output_dir).resolve()
    dom_sub = {"wall": "wall_mounted", "furniture": "furniture",
               "decoration": "decorations"}[domain]
    dom_dir = output_dir / dom_sub
    boxes_json = dom_dir / "_la_boxes.json"
    if not boxes_json.exists():
        raise FileNotFoundError(f"no boxes to segment: {boxes_json}")
    _run([_env_python(_SAM2_ENV), "sam2_from_boxes.py",
          "--boxes", str(boxes_json), "--sam-type", _SAM2_TYPE,
          "--contract", "--output-dir", str(output_dir), "--domain", domain],
         cwd=_SAM2_DIR)
    print(f"[la-sam2] {domain} re-segmented from boxes → "
          f"{dom_dir}/segment_results.json")
    return dom_dir

