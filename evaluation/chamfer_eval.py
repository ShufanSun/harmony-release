"""Chamfer + F-Score eval using Gen3DSR's exact `compute_metrics_meshes` function.

Their formula (https://github.com/AndreeaDogaru/Gen3DSR/blob/main/src/eval_front.py):

    def compute_metrics_meshes(gt, pred, num_points=1000000, thresholds=[0.1, 0.01, 0.001], eps=1e-6):
        gt_points   = gt.as_open3d.sample_points_uniformly(num_points)
        pred_points = pred.as_open3d.sample_points_uniformly(num_points)
        dist_gt_pred = np.array(gt_points.compute_point_cloud_distance(pred_points))
        dist_pred_gt = np.array(pred_points.compute_point_cloud_distance(gt_points))
        metrics["Chamfer"] = (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2
        for t in thresholds:
            precision = 100.0 * (dist_pred_gt < t).mean()
            recall    = 100.0 * (dist_gt_pred < t).mean()
            f1 = 2 * precision * recall / (precision + recall + eps)
            ...

Our adaptation:
  - Use their function verbatim
  - Pre-process: normalize each pc to unit cube + ICP-align method → GT (because
    methods aren't in GT's FRONT3D world frame; without alignment metrics are
    meaningless)
  - 30k sample points instead of 1M (fits in GPU memory; 1M would OOM on
    pytorch3d ICP)
  - Skip frustum culling (their code uses GT camera intrinsics on pred meshes
    which assumes pred is already in GT camera frame — none of our methods are)

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 python chamfer_eval_paper.py \\
        --gt-root /path/to/front3d_gt/sceneobjgt \\
        --out results_harmony300_chamfer_paper.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import trimesh
from pytorch3d.loss import chamfer_distance as p3d_chamfer
from pytorch3d.ops import iterative_closest_point
from tqdm import tqdm

N_SAMPLES = 30000
THRESHOLDS = [0.1, 0.01, 0.001]


# ─── their function, verbatim except num_points + skipping `.as_open3d` ─────


def compute_metrics_pc(gt_o3d_pc, pred_o3d_pc, thresholds=THRESHOLDS, eps=1e-6):
    """Verbatim Gen3DSR chamfer / F-Score on already-sampled open3d PCs."""
    metrics = {}
    dist_gt_pred = np.array(gt_o3d_pc.compute_point_cloud_distance(pred_o3d_pc))
    dist_pred_gt = np.array(pred_o3d_pc.compute_point_cloud_distance(gt_o3d_pc))
    metrics["Chamfer"] = (dist_gt_pred.mean() + dist_pred_gt.mean()) / 2

    for t in thresholds:
        precision = 100.0 * (dist_pred_gt < t).mean()
        recall = 100.0 * (dist_gt_pred < t).mean()
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics[f"Precision@{t}"] = float(precision)
        metrics[f"Recall@{t}"] = float(recall)
        metrics[f"F1@{t}"] = float(f1)
    metrics["Chamfer"] = float(metrics["Chamfer"])
    return metrics


# ─── point cloud loading ────────────────────────────────────────────────────


_ROOM_PARTS = ("floor", "ceiling", "wall")


def sample_pc_from_mesh(path: str, n: int = N_SAMPLES, exclude_room=False):
    try:
        m = trimesh.load(path, force=None if exclude_room else "mesh")
    except Exception as e:
        print(f"  load fail: {path} ({e})", file=sys.stderr)
        return None
    if isinstance(m, trimesh.Scene):
        if exclude_room:
            meshes = [g for k, g in m.geometry.items()
                      if isinstance(g, trimesh.Trimesh)
                      and not any(k.lower().startswith(p) for p in _ROOM_PARTS)]
        else:
            meshes = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return None
        m = trimesh.util.concatenate(meshes)
    if not hasattr(m, "sample") or getattr(m, "area", 0) == 0:
        verts = np.asarray(getattr(m, "vertices", []))
        if len(verts) == 0:
            return None
        idx = np.random.choice(len(verts), min(n, len(verts)), replace=len(verts) < n)
        return verts[idx]
    pts, _ = trimesh.sample.sample_surface(m, n)
    return np.asarray(pts)


def load_ply_pc(path: str, n: int = N_SAMPLES):
    try:
        pc = trimesh.load(path)
    except Exception as e:
        print(f"  load fail: {path} ({e})", file=sys.stderr)
        return None
    verts = np.asarray(pc.vertices)
    if len(verts) == 0:
        return None
    if len(verts) > n:
        idx = np.random.choice(len(verts), n, replace=False)
        verts = verts[idx]
    return verts


def load_gen3dsr_pc(scene, root, n=N_SAMPLES):
    d = os.path.join(root, scene, "object_space")
    if not os.path.isdir(d):
        return None
    plys = sorted(f for f in os.listdir(d) if f.endswith(".ply"))
    if not plys:
        return None
    per = max(n // len(plys), 200)
    out = []
    for p in plys:
        pc = load_ply_pc(os.path.join(d, p), per)
        if pc is not None:
            out.append(pc)
    return np.vstack(out) if out else None


def load_3dregen_pc(scene, root, n=N_SAMPLES):
    p = os.path.join(root, scene, "combined_scene.glb")
    return sample_pc_from_mesh(p, n) if os.path.isfile(p) else None


def load_harmony_pc(scene, root, n=N_SAMPLES):
    for sub in ("furniture/scene_with_furniture.glb",
                "ceiling/scene_with_ceiling.glb",
                "ceiling/scene_full_with_ceiling.glb"):
        p = os.path.join(root, scene, sub)
        if os.path.isfile(p):
            return sample_pc_from_mesh(p, n, exclude_room=True)
    return None


def load_sam3d_pc(scene, root, n=N_SAMPLES):
    # assembled_scene.glb = furniture-only (no walls/ceiling/floor), matching
    # FRONT3D GT. assembled_scene_room.glb ADDS room geometry so is not used.
    p = os.path.join(root, scene, "assembled_scene.glb")
    return sample_pc_from_mesh(p, n) if os.path.isfile(p) else None


# CAST part-name substrings to drop so only furniture remains (GT is
# furniture-only). CAST parts are semantically named, e.g. object_1_wall,
# object_11_hardwood_floor_floor, object_10_curtain, object_12_window, floor.
_CAST_ROOM_PARTS = ("wall", "floor", "carpet", "curtain", "window", "ceiling", "door")


def load_cast_pc(scene, root, n=N_SAMPLES):
    # {root}/{scene}/{scene}/reconstructed_scene_room.glb (double-nested).
    # Local re-runs only emit reconstructed_scene.glb (no _room); both give
    # identical furniture geometry after room-part filtering, so fall back.
    p = os.path.join(root, scene, scene, "reconstructed_scene_room.glb")
    if not os.path.isfile(p):
        p = os.path.join(root, scene, scene, "reconstructed_scene.glb")
    if not os.path.isfile(p):
        return None
    try:
        s = trimesh.load(p)
    except Exception as e:
        print(f"  load fail: {p} ({e})", file=sys.stderr)
        return None
    if isinstance(s, trimesh.Scene):
        keep = [g for k, g in s.geometry.items()
                if isinstance(g, trimesh.Trimesh)
                and not any(w in k.lower() for w in _CAST_ROOM_PARTS)]
        if not keep:
            return None
        s = trimesh.util.concatenate(keep)
    if not hasattr(s, "sample") or getattr(s, "area", 0) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(s, n)
    return np.asarray(pts)


def load_viga_pc(scene, root, n=N_SAMPLES):
    # {root}/{scene}/scene_objects.glb, exported from VIGA's blender_file.blend by
    # VIGA/results/export_viga_objects.py: render-visible MESH objects only
    # (hide_render ones are objects the agent disabled), minus the room shell
    # (same floor/ceiling/wall rule as _ROOM_PARTS). Node transforms are baked by
    # force="mesh" inside sample_pc_from_mesh; Y-up like the other methods' GLBs.
    p = os.path.join(root, scene, "scene_objects.glb")
    if not os.path.isfile(p):
        return None
    return sample_pc_from_mesh(p, n)


# Astra Harmony release (astra-results/harmony300_meshes_glb): scenes are
# semantically named (bedroom_01, living_room_03, ...) and each scene.glb bundles
# the full room shell + window backdrop (sky + distant city buildings) + furniture.
# GT is furniture-only, so drop the architectural shell, the environment backdrop,
# and architectural door/window hardware — but KEEP furniture (incl. "Cabinet door").
_ASTRA_EXCLUDE = (
    "floor", "ceiling", "wall", "baseboard", "soffit", "window", "muntin",
    "sill", "cornice", "molding", "skirting", "trim",
    "sky", "distant", "city", "building", "backdrop", "skyline", "cityscape",
    "horizon", "environment", "daylight", "sun",
    "door leaf", "door lever", "door handle", "door recessed panel", "door frame",
    "door hinge", "door rose", "door lock", "door stile", "door rail", "door panel",
    # tall architectural openings + ceiling fixtures GT (floor-furniture only)
    # lacks; kept, they inflate one axis and wreck the per-scene normalize.
    "casing", "chandelier", "pendant", "luminaire", "pelmet", "valance",
    # small tabletop / ceiling decor GT does not model (books, vases, plants,
    # lamp/glass shades, ornaments) — filtered so astra is furniture-only like
    # GT and the old Harmony meshes, for a like-for-like chamfer.
    "shade", "vase", "plant", "potted", "flower", "greenery", "foliage",
    "bottle", "ornament", "figurine", "sculpture", "bowl", "jar", "decor",
    # thin wall-attached trim that survives the slab filters yet spans a whole
    # axis (near-zero area, huge extent): wall-panel seams / rails, picture
    # frames, wall art. Phrases, not bare "frame"/"panel"/"rail", so sofa/bed
    # frames, cabinet panels and bed rails (real furniture) are kept.
    "panel joint", "panel_joint", "paneling", "panelling", "wainscot",
    "frame vertical rail", "frame horizontal rail", "framed surface",
    "print frame", "picture frame", "photo frame", "poster", "wall art",
    "artwork", "painting", "bust",
)
# Set $ASTRA_ATTRIB to the benchmark's attribution.csv (rgb_ id -> scene name).
_ASTRA_ATTRIB_DEFAULT = os.environ.get("ASTRA_ATTRIB", "")
_astra_rgb2name_cache = None


def _astra_rgb2name():
    """Map front3d rgb_ id -> astra semantic name via attribution.csv (easy split).

    The 5th column (mislabelled 'pexels_id') actually holds the rgb_ id for the
    easy split, whose source is 3D-FRONT. Cached after first read.
    """
    global _astra_rgb2name_cache
    if _astra_rgb2name_cache is None:
        import csv
        attrib = os.environ.get("ASTRA_ATTRIB", _ASTRA_ATTRIB_DEFAULT)
        m = {}
        if os.path.isfile(attrib):
            for r in csv.DictReader(open(attrib)):
                if r.get("difficulty") == "easy":
                    m[r["pexels_id"]] = r["name"]
        _astra_rgb2name_cache = m
    return _astra_rgb2name_cache


def load_harmony_astra_pc(scene, root, n=N_SAMPLES):
    name = _astra_rgb2name().get(scene)
    if name is None:
        return None
    p = None
    for sub in (f"easy/{name}/scene.glb", f"{name}/scene.glb"):
        cand = os.path.join(root, sub)
        if os.path.isfile(cand):
            p = cand
            break
    if p is None:
        return None
    try:
        s = trimesh.load(p)
    except Exception as e:
        print(f"  load fail: {p} ({e})", file=sys.stderr)
        return None
    if isinstance(s, trimesh.Scene):
        keep = [g for k, g in s.geometry.items()
                if isinstance(g, trimesh.Trimesh)
                and not any(w in k.lower() for w in _ASTRA_EXCLUDE)]
        if not keep:
            return None
        s = trimesh.util.concatenate(keep)
    if not hasattr(s, "sample") or getattr(s, "area", 0) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(s, n)
    return np.asarray(pts)


METHOD_LOADERS = {
    "Gen3DSR": load_gen3dsr_pc,
    "3DREGEN": load_3dregen_pc,
    "Harmony": load_harmony_pc,
    "Harmony-astra": load_harmony_astra_pc,
    "SAM3D": load_sam3d_pc,
    "CAST": load_cast_pc,
    "VIGA": load_viga_pc,
}


# ─── normalize + ICP (pre-process for paper formula) ────────────────────────


def normalize_pc(pc):
    centered = pc - pc.mean(axis=0)
    scale = max((pc.max(axis=0) - pc.min(axis=0)).max(), 1e-9)
    return centered / scale


def pca_rot(pc_np):
    c = pc_np.mean(axis=0)
    centered = pc_np - c
    cov = (centered.T @ centered) / max(len(pc_np), 1)
    _, vecs = np.linalg.eigh(cov)
    R = vecs[:, ::-1]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def icp_align(src_np, dst_np, device, max_iter=30):
    """Return best-aligned src as numpy (try 4 PCA-axis-flip inits)."""
    R_src = pca_rot(src_np).T
    R_dst = pca_rot(dst_np)
    flips = [np.eye(3), np.diag([1.0, -1.0, -1.0]),
             np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])]
    dst_t = torch.from_numpy(dst_np).float().unsqueeze(0).to(device)
    src_t_orig = torch.from_numpy(src_np).float().to(device)
    best_cd_sq = float("inf")
    best_aligned_np = None
    for flip in flips:
        R_init = R_dst @ flip @ R_src
        R_init_t = torch.from_numpy(R_init).float().to(device)
        src_t = (src_t_orig @ R_init_t.T).unsqueeze(0)
        result = iterative_closest_point(src_t, dst_t, max_iterations=max_iter,
                                          relative_rmse_thr=1e-6)
        aligned = result.Xt
        cd_sq, _ = p3d_chamfer(aligned, dst_t)
        v = float(cd_sq.item())
        if v < best_cd_sq:
            best_cd_sq = v
            best_aligned_np = aligned.squeeze(0).cpu().numpy()
    return best_aligned_np


def np_to_o3d_pc(pts: np.ndarray):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    return pc


# ─── driver ─────────────────────────────────────────────────────────────────


def process_one(scene, gt_dir, methods, device, do_icp=True):
    gt_id = scene.split("_")[1] if "_" in scene else scene
    gt_path = os.path.join(gt_dir, f"sceneobjgt_{gt_id}.ply")
    gt_raw = load_ply_pc(gt_path)
    if gt_raw is None:
        return {"scene": scene, "error": f"no GT at {gt_path}"}
    gt_norm = normalize_pc(gt_raw)
    gt_o3d = np_to_o3d_pc(gt_norm)
    out = {"scene": scene}
    for method, root in methods.items():
        loader = METHOD_LOADERS[method]
        pc = loader(scene, root)
        if pc is None:
            out[method] = None
            continue
        try:
            pc_norm = normalize_pc(pc)
            if do_icp:
                aligned = icp_align(pc_norm, gt_norm, device)
            else:
                aligned = pc_norm
            pred_o3d = np_to_o3d_pc(aligned)
            out[method] = compute_metrics_pc(gt_o3d, pred_o3d)
        except Exception as e:
            msg = f"{type(e).__name__}:{e}"
            out[method] = {"error": msg}
            # An uncorrectable ECC fault latches the GPU dead: every later scene
            # will fail the same way. Bail out instead of burning the whole
            # walltime producing a file full of identical errors.
            if "ECC" in msg or "CUDA error" in msg:
                if "no kernel image" in msg:
                    why = ("this env's torch/pytorch3d has no kernels for this GPU's "
                           "compute capability -- pick a newer card, not a new env")
                elif "ECC" in msg:
                    why = "uncorrectable ECC: the card is latched dead, resubmit elsewhere"
                else:
                    why = "unrecoverable CUDA fault, resubmit elsewhere"
                raise SystemExit(
                    f"[abort] GPU fault on {scene}/{method}: {msg.splitlines()[0]}\n"
                    f"[abort] {why}."
                )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-root", required=True)
    p.add_argument("--gen3dsr-root",
                   default=None)
    p.add_argument("--regen-root",
                   default=None)
    p.add_argument("--harmony-root",
                   default=None)
    p.add_argument("--sam3d-root",
                   default=None)
    p.add_argument("--cast-root",
                   default=None)
    p.add_argument("--viga-root", default=None,
                   help="VIGA GLB root: {root}/{scene}/scene_objects.glb")
    p.add_argument("--harmony-astra-root", default=None,
                   help="Astra Harmony release root: {root}/easy/{semantic}/scene.glb "
                        "(rgb_ id mapped via attribution.csv; set $ASTRA_ATTRIB to override)")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of methods to run (e.g. SAM3D)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-icp", action="store_true",
                   help="skip ICP alignment (just normalize, like paper assumes pred in GT frame)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    methods = {
        "Gen3DSR": args.gen3dsr_root,
        "3DREGEN": args.regen_root,
        "Harmony": args.harmony_root,
        "Harmony-astra": args.harmony_astra_root,
        "SAM3D": args.sam3d_root,
        "CAST": args.cast_root,
        "VIGA": args.viga_root,
    }
    if args.only:
        keep = {m.strip() for m in args.only.split(",")}
        methods = {m: r for m, r in methods.items() if m in keep}
    methods = {m: r for m, r in methods.items() if r}   # drop methods with no root given

    # Scene list = exactly the GT we have on disk, not whatever one baseline's
    # output dir happens to contain.
    scenes = sorted(
        "rgb_" + f[len("sceneobjgt_"):-len(".ply")]
        for f in os.listdir(args.gt_root)
        if f.startswith("sceneobjgt_") and f.endswith(".ply")
    )
    print(f"  scenes: {len(scenes)}", flush=True)

    t0 = time.time()
    results = []
    for s in tqdm(scenes, desc="chamfer-paper"):
        results.append(process_one(s, args.gt_root, methods, device, do_icp=not args.no_icp))
    print(f"  total: {(time.time() - t0)/60:.1f} min  (icp={'off' if args.no_icp else 'on'})",
          flush=True)

    metric_keys = (["Chamfer"]
                   + [f"{p}@{t}" for t in THRESHOLDS for p in ("Precision", "Recall", "F1")])
    summary = {}
    for m in methods:
        per_metric = {}
        for k in metric_keys:
            vals = [r[m][k] for r in results
                    if isinstance(r.get(m), dict) and isinstance(r[m].get(k), (int, float))]
            per_metric[k] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "median": float(np.median(vals)) if vals else None,
            }
        summary[m] = per_metric

    payload = {"summary": summary, "per_scene": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"{'method':<10} {'Chamfer↓':>10} {'F1@0.1↑':>10} {'F1@0.01↑':>10} {'F1@0.001↑':>11}")
    for m in methods:
        s = summary[m]
        if s["Chamfer"]["mean"] is None:
            print(f"  {m:<8} {'— no scenes scored —':>44}")
            continue
        print(f"  {m:<8} "
              f"{s['Chamfer']['mean']:>10.4f} "
              f"{s['F1@0.1']['mean']:>10.2f} "
              f"{s['F1@0.01']['mean']:>10.2f} "
              f"{s['F1@0.001']['mean']:>11.3f}")


if __name__ == "__main__":
    main()
