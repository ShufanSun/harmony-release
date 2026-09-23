"""
batch_main.py — run the full main.py pipeline on every image in an input dir.

Subprocess-calls `python main.py --image <img> --output outputs/<stem>` for
each photo, so it stays in sync with main.py's stage list automatically.

Default input is `data/indoor_images/`.  Per-scene output goes
to `outputs/<image_stem>/`.

A scene is considered DONE (skipped on re-run) when its output directory
contains `decorations/placements/decoration_placements.json` — the final
stage's output.  Use --force to re-run completed scenes.

Per-scene stdout/stderr is mirrored to `outputs/<stem>/batch_run.log`.
Failures don't kill subsequent images — they're logged and processing
continues.

Usage:
    # All images in data/indoor_images/, all 16 stages:
    python scripts/batch_main.py

    # Only specific images:
    python scripts/batch_main.py --only pexels_1571468 pexels_1743229

    # Skip the slow 3D-gen + placement stages while iterating on earlier ones:
    python scripts/batch_main.py --skip object_generation \
        furniture_object_generation decoration_object_generation \
        place_decorations

    # Different input/output dirs:
    python scripts/batch_main.py \
        --input-dir data/my_photos \
        --output-root outputs/Batch

    # Re-run scenes that already finished:
    python scripts/batch_main.py --force
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
_DEFAULT_INPUT_DIR   = REPO_ROOT / "data" / "indoor_images"
_DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
_DONE_MARKER = "decorations/placements/decoration_placements.json"


def _gather_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        sys.exit(f"input dir does not exist: {input_dir}")
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _scene_done(scene_dir: Path) -> bool:
    return (scene_dir / _DONE_MARKER).is_file()


def _run_one(image: Path, scene_dir: Path,
             skip: list[str], stages: list[str] | None,
             rerun: bool) -> tuple[bool, float]:
    """Invoke main.py for one image, mirroring output to a log file.
    Returns (success, elapsed_seconds)."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    log_path = scene_dir / "batch_run.log"

    cmd = [sys.executable, "main.py",
           "--image", str(image)]
    if rerun:
        cmd += ["--rerun-dir", str(scene_dir)]
    else:
        cmd += ["--output", str(scene_dir)]
    if stages:
        cmd += ["--stages", *stages]
    if skip:
        cmd += ["--skip", *skip]

    print(f"[batch] $ {' '.join(shlex.quote(c) for c in cmd)}")
    start = time.time()
    with open(log_path, "ab") as logfp:
        header = (f"\n{'═'*70}\n"
                  f" batch_main.py — {datetime.now().isoformat(timespec='seconds')}\n"
                  f" image: {image}\n"
                  f" cmd:   {' '.join(shlex.quote(c) for c in cmd)}\n"
                  f"{'═'*70}\n").encode()
        logfp.write(header)
        logfp.flush()
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            stdout=logfp, stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    elapsed = time.time() - start
    return (proc.returncode == 0), elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(_DEFAULT_INPUT_DIR),
                    help=f"Directory of input images "
                         f"(default {_DEFAULT_INPUT_DIR.relative_to(REPO_ROOT)}).")
    ap.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT),
                    help=f"Per-scene output goes to <output-root>/<image_stem>/  "
                         f"(default {_DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT)}).")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only process images whose stem matches one of these "
                         "names (no extension).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run scenes that already have the done marker.")
    ap.add_argument("--rerun", action="store_true",
                    help="Pass --rerun-dir to main.py instead of --output, so "
                         "main.py reuses any depth/analysis/textures already on "
                         "disk in the scene dir.  Implies the dir already exists.")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="Forward to main.py --stages.")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="Forward to main.py --skip.")
    args = ap.parse_args()

    input_dir   = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = _gather_images(input_dir)
    if args.only:
        wanted = set(args.only)
        images = [p for p in images if p.stem in wanted]
        missing = wanted - {p.stem for p in images}
        if missing:
            print(f"[batch] WARNING: --only listed unknown stems: {sorted(missing)}")

    if not images:
        sys.exit(f"[batch] no images to process under {input_dir}")

    print(f"[batch] {len(images)} image(s) under {input_dir.relative_to(REPO_ROOT)}/")
    print(f"[batch] output root: {output_root.relative_to(REPO_ROOT)}/")
    if args.skip:    print(f"[batch] skipping stages: {args.skip}")
    if args.stages:  print(f"[batch] running only stages: {args.stages}")

    summary: list[tuple[str, str, float]] = []  # (stem, status, secs)
    t_total = time.time()
    for i, img in enumerate(images, 1):
        scene_dir = output_root / img.stem
        print(f"\n{'═'*70}\n [{i}/{len(images)}] {img.name}  → {scene_dir.relative_to(REPO_ROOT)}/\n{'═'*70}")

        if not args.force and _scene_done(scene_dir):
            print(f"[batch] SKIP — already done (found {_DONE_MARKER})")
            summary.append((img.stem, "skipped", 0.0))
            continue

        try:
            ok, secs = _run_one(
                image=img, scene_dir=scene_dir,
                skip=args.skip, stages=args.stages,
                rerun=args.rerun,
            )
            status = "ok" if ok else "FAIL"
            print(f"[batch] {status} in {secs:.1f}s  "
                  f"(log: {(scene_dir / 'batch_run.log').relative_to(REPO_ROOT)})")
            summary.append((img.stem, status, secs))
        except KeyboardInterrupt:
            print(f"\n[batch] interrupted by user — stopping.")
            break
        except Exception as e:
            print(f"[batch] EXCEPTION running {img.name}: {e}")
            summary.append((img.stem, "EXCEPTION", 0.0))

    print(f"\n{'═'*70}\n BATCH SUMMARY  ({time.time()-t_total:.1f}s total)\n{'═'*70}")
    for stem, status, secs in summary:
        print(f"  {status:<10}  {secs:>6.1f}s  {stem}")
    n_ok    = sum(1 for _, s, _ in summary if s == "ok")
    n_fail  = sum(1 for _, s, _ in summary if s == "FAIL")
    n_skip  = sum(1 for _, s, _ in summary if s == "skipped")
    n_exc   = sum(1 for _, s, _ in summary if s == "EXCEPTION")
    print(f"\n  ok={n_ok}  fail={n_fail}  exception={n_exc}  skipped={n_skip}")


if __name__ == "__main__":
    main()
