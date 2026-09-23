"""Eval runner: compute reference-based metrics across multiple methods.

Per-method ``loaders`` (see ``loaders.py``) build ``{key: image_path}`` maps;
the runner then matches predictions to the reference set by shared key and
computes per-pair + averaged CLIP-cosine-distance and photometric MSE via
``metrics.py``.

Usage:
    python eval_runner.py \
        --ref flat:/path/to/harmony300/ref \
        --pred 3DREGEN=3dregen:/path/to/3D-RE-GEN/results \
        --pred SceneGen=scenegen:/path/to/SceneGen/renders \
        --out results.json

Each ``--ref`` / ``--pred`` value uses ``LOADER:DIR`` (and ``--pred`` also takes
a ``NAME=`` prefix).  Available loaders: see ``loaders.LOADERS``.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

from PIL import Image
from tqdm import tqdm

try:
    import pillow_avif  # registers AVIF support with PIL on import
except ImportError:
    pass

from loaders import get_loader
from metrics import (
    clip_similarity,
    ensure_clip_loaded,
    ensure_lpips_loaded,
    lpips_distance,
    photometric_loss,
)


def parse_loader_spec(spec: str) -> tuple[str, str]:
    """Parse ``LOADER:DIR`` into ``(loader_name, dir_path)``."""
    if ":" not in spec:
        raise SystemExit(f"expected LOADER:DIR, got: {spec!r}")
    loader_name, dir_path = spec.split(":", 1)
    return loader_name.strip(), os.path.expanduser(dir_path.strip())


def parse_pred_spec(spec: str) -> tuple[str, str, str]:
    """Parse ``NAME=LOADER:DIR`` into ``(name, loader_name, dir_path)``."""
    if "=" not in spec:
        raise SystemExit(f"--pred expects NAME=LOADER:DIR, got: {spec!r}")
    name, rest = spec.split("=", 1)
    loader_name, dir_path = parse_loader_spec(rest)
    return name.strip(), loader_name, dir_path


def evaluate_method(
    name: str,
    pred_index: dict[str, str],
    ref_index: dict[str, str],
) -> dict:
    """Run metrics for every (pred, ref) pair sharing a key."""
    if not pred_index:
        print(f"[WARN] {name}: no images indexed", file=sys.stderr)
        return {"num_pairs": 0, "per_scene": {}, "averages": {}}

    shared_keys = sorted(set(pred_index) & set(ref_index))
    missing_ref = sorted(set(pred_index) - set(ref_index))
    missing_pred = sorted(set(ref_index) - set(pred_index))

    per_scene: dict[str, dict] = {}
    n_clip_vals: list[float] = []
    pl_vals: list[float] = []
    lpips_vals: list[float] = []

    for key in tqdm(shared_keys, desc=f"  {name}"):
        try:
            pred_img = Image.open(pred_index[key]).convert("RGB")
            ref_img = Image.open(ref_index[key]).convert("RGB")
        except Exception as e:
            per_scene[key] = {"error": f"{type(e).__name__}: {e}"}
            continue

        size_mismatch = pred_img.size != ref_img.size
        if size_mismatch:
            print(
                f"[WARN] {name} {key}: size mismatch pred={pred_img.size} "
                f"ref={ref_img.size} (resized inside metrics — pl unreliable)",
                file=sys.stderr,
            )

        try:
            n_clip = float(1 - clip_similarity(pred_img, ref_img))
            pl = float(photometric_loss(pred_img, ref_img))
            lp = float(lpips_distance(pred_img, ref_img))
        except Exception as e:
            per_scene[key] = {"error": f"{type(e).__name__}: {e}"}
            continue

        entry = {"n_clip": n_clip, "pl": pl, "lpips": lp}
        if size_mismatch:
            entry["size_mismatch"] = {
                "pred": list(pred_img.size),
                "ref": list(ref_img.size),
            }
        per_scene[key] = entry
        n_clip_vals.append(n_clip)
        pl_vals.append(pl)
        lpips_vals.append(lp)

    averages = {}
    if n_clip_vals:
        averages = {
            "avg_n_clip": sum(n_clip_vals) / len(n_clip_vals),
            "avg_pl": sum(pl_vals) / len(pl_vals),
            "avg_lpips": sum(lpips_vals) / len(lpips_vals),
            "num_scored": len(n_clip_vals),
        }

    return {
        "num_pred": len(pred_index),
        "num_pairs": len(shared_keys),
        "num_missing_ref": len(missing_ref),
        "num_missing_pred": len(missing_pred),
        "missing_ref_sample": missing_ref[:5],
        "missing_pred_sample": missing_pred[:5],
        "averages": averages,
        "per_scene": per_scene,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ref",
        required=True,
        help="reference set as LOADER:DIR (e.g. 'flat:/.../testing_first1')",
    )
    p.add_argument(
        "--pred",
        action="append",
        default=[],
        help="NAME=LOADER:DIR for a method (can be repeated)",
    )
    p.add_argument("--out", default="eval_results.json", help="Output JSON file")
    args = p.parse_args()

    ref_loader_name, ref_dir = parse_loader_spec(args.ref)
    if not os.path.isdir(ref_dir):
        raise SystemExit(f"reference dir not found: {ref_dir}")
    ref_index = get_loader(ref_loader_name)(ref_dir)
    print(f"Reference: {ref_loader_name}:{ref_dir}  →  {len(ref_index)} items")
    if not ref_index:
        raise SystemExit("Reference index is empty.")

    if not args.pred:
        raise SystemExit("provide at least one --pred NAME=LOADER:DIR")

    methods = [parse_pred_spec(s) for s in args.pred]
    ensure_clip_loaded()  # warm up once
    ensure_lpips_loaded()

    results: dict = {
        "ref_dir": ref_dir,
        "ref_loader": ref_loader_name,
        "num_ref": len(ref_index),
        "methods": OrderedDict(),
    }

    for name, loader_name, pred_dir in methods:
        # Some loaders (e.g. ``harmony``) accept comma-separated multi-root paths,
        # so let the loader itself decide whether the dir is valid. We only warn
        # if a single-root path doesn't exist (and even then, still call the loader
        # — it'll return an empty index and the method will be reported with 0 pairs).
        if "," not in pred_dir and not os.path.isdir(pred_dir):
            print(f"[WARN] {name}: dir not found: {pred_dir}", file=sys.stderr)
        pred_index = get_loader(loader_name)(pred_dir)
        print(f"\n=== {name} ===  ({loader_name}:{pred_dir})  →  {len(pred_index)} preds")
        method_result = evaluate_method(name, pred_index, ref_index)
        method_result["loader"] = loader_name
        method_result["dir"] = pred_dir
        results["methods"][name] = method_result

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'method':<20} {'pairs':>6} {'n_clip':>10} {'pl':>10} {'lpips':>10}")
    for name, r in results["methods"].items():
        avg = r.get("averages", {})
        if avg:
            print(
                f"{name:<20} {r['num_pairs']:>6} "
                f"{avg['avg_n_clip']:>10.4f} {avg['avg_pl']:>10.4f} "
                f"{avg.get('avg_lpips', float('nan')):>10.4f}"
            )
        else:
            print(f"{name:<20} {'-':>6} {'-':>10} {'-':>10} {'-':>10}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
