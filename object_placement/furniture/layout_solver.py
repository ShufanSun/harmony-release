"""Joint-energy layout solver — A/B prototype.

Runs AFTER the existing placement cascade and re-solves every object's
(x, z, yaw, scale) *simultaneously* against one scalar energy, instead of
mutating them through ~10 sequential heuristic passes.

Modelled on VIGA / the generator's `run_render_compare`
(VIGA-release/utils/third_party/the 3D generator/.../layout_post_optimization_utils.py),
which minimises

    w_mask·MSE(render, mask_gt) + w_q·||q-I||² + w_t·||t||² + w_s·(s-1)²

in two stages (translation+scale, then unlock rotation) with an explicit
ΔLoss convergence test.  Three properties of that design are what the
current cascade lacks:

  1. ONE objective.  Passes cannot silently undo each other, because they
     are simultaneous terms rather than sequential writes.  (office8:
     mask_align dragged the facing-placed chair back 1.34 m; `facing` and
     `mask_align` are now competing terms and the optimum is the trade-off.)
  2. A PRIOR REGULARISER.  In the cascade a 1.34 m drag or a 1 m
     vlm_layout relocation costs exactly nothing.  Here every metre of
     deviation from the pipeline's estimate is paid for.
  3. PRIORITY AS A NUMBER.  "wall affinity should not override vlm
     reflection" becomes w_wall vs w_face, not statement ordering.

To that we add the two terms VIGA has no analogue for, because they do not
decompose per-object — VIGA aligns each mesh independently:

  * WALL SEMANTICS  E_wall   — wall_affinity as a distance-to-wall-plane cost
  * GLOBAL LAYOUT   E_coll   — pairwise penetration, replacing push/clip/vlm_layout
                    E_depth  — the `depth` ordering the cascade never reads at all

Deliberately derivative-free: E_sil uses the existing (non-differentiable)
projection, and 6 objects x 4 DoF = 24 dims is well inside Powell's range.
Swapping in a differentiable silhouette (pytorch3d is already vendored under
third_party/the 3D generator) is the upgrade path if the silhouette fit is too loose.

Geometry note: every term is evaluated on TRANSFORMED MESH VERTICES, never on
cached size_m scalars.  That is deliberate — the width_m/depth_m-vs-raw-axis
confusion that caused the office8 desk bug cannot occur here by construction.

Usage:
    python layout_solver.py <scene_dir> [--out NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize

# ── Energy weights ───────────────────────────────────────────────────────
# Ratios matter more than absolutes.  VIGA uses mask:reg ~ 2000-4000x; we sit
# lower because our prior is the pipeline's converged answer (a much stronger
# starting point than VIGA's identity transform), so it deserves more trust.
W = {
    "sil":   60.0,   # silhouette IoU vs the segmentation bbox
    "wall":  40.0,   # wall_affinity flush
    "face":  25.0,   # facing_toward alignment
    "coll": 120.0,   # interpenetration — the hardest constraint
    "depth":  8.0,   # VLM depth ordering within a group
    "room":  80.0,   # stay inside the walls
    "prior":  6.0,   # deviation from the pipeline estimate
}

# Prior normalisation: how far an object may drift before it costs ~1.0
PRIOR_XZ_M   = 0.30
PRIOR_YAW_RAD = math.radians(20.0)
PRIOR_LOGS   = math.log(1.15)

WALL_GAP   = 0.01
DEPTH_MARGIN = 0.12     # metres of z-separation a depth-ordering pair should keep
MAX_VERTS  = 400        # subsample per mesh — plenty for bbox/hull/extents


# ── Scene loading ────────────────────────────────────────────────────────

def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _yaw_of(R: np.ndarray) -> float:
    """Extract the Y-rotation of a (near-)yaw-only matrix."""
    return math.atan2(float(R[0][2]), float(R[0][0]))


class Camera:
    """Pinhole projection matching the pipeline's _backproject_pixel frame."""

    def __init__(self, cam: dict):
        self.pos = np.array(cam["position_m"], dtype=np.float64)
        look_at  = np.array(cam["look_at_m"], dtype=np.float64)
        up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
        self.W = float(cam["width_px"])
        self.H = float(cam["height_px"])
        fwd = look_at - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up_world)
        right /= np.linalg.norm(right)
        up_c = np.cross(right, fwd)
        self.fwd, self.right, self.up = fwd, right, up_c
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.fx = (self.W / 2.0) / math.tan(math.radians(float(cam["hfov_deg"])) / 2.0)

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World (N,3) -> pixel (M,2), dropping points behind the camera."""
        d = pts - self.pos
        z = d @ self.fwd
        ok = z > 1e-6
        if not np.any(ok):
            return np.empty((0, 2))
        d, z = d[ok], z[ok]
        u = self.cx + self.fx * (d @ self.right) / z
        v = self.cy - self.fx * (d @ self.up) / z
        return np.stack([u, v], axis=1)


class Obj:
    """One solvable placement: its mesh, its constraints, its initial pose."""

    def __init__(self, p: dict, entry: dict, scene_dir: Path):
        self.idx  = p.get("index")
        self.type = (p.get("type") or entry.get("type") or "").lower()
        self.p    = p

        # Initial (pipeline) pose — the prior.
        self.pos0   = np.array(p["position_m"], dtype=np.float64)
        R0          = np.array(p["rotation_3x3"], dtype=np.float64)
        self.yaw0   = _yaw_of(R0)
        sc          = np.asarray(p.get("scale", [1, 1, 1]), dtype=np.float64).reshape(-1)
        self.scale0 = np.repeat(sc, 3)[:3] if sc.size == 1 else sc

        # Semantics from the analysis entry.
        self.wall    = p.get("wall_affinity", entry.get("wall_affinity", "centre"))
        self.facing  = entry.get("facing_toward")
        self.group   = entry.get("group")
        self.depth_tag = (entry.get("depth") or "").lower()
        self.box_px  = entry.get("box_px")

        # Local mesh verts, centred the same way the pipeline centres them.
        # glb_path may be absolute, repo-root-relative, or scene-relative.
        raw = p.get("glb_path") or ""
        for cand in (Path(raw), Path.cwd() / raw, scene_dir / raw,
                     scene_dir / "furniture" / Path(raw).name,
                     scene_dir / "furniture" / "objects" / Path(raw).name):
            if cand.is_file():
                gp = cand
                break
        else:
            raise FileNotFoundError(raw)
        m = trimesh.load(str(gp), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(m.dump())
        v = np.asarray(m.vertices, dtype=np.float64)
        if len(v) > MAX_VERTS:
            v = v[np.random.default_rng(0).choice(len(v), MAX_VERTS, replace=False)]
        self.v_local = v - np.asarray(m.vertices, dtype=np.float64).mean(axis=0)

        lf = np.array(p.get("_local_front", [0, 0, -1]), dtype=np.float64)
        lf[1] = 0.0
        n = np.linalg.norm(lf)
        self.local_front = lf / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])

    # -- pose application ------------------------------------------------
    def world_verts(self, dx: float, dz: float, dyaw: float, logs: float) -> np.ndarray:
        R = _rot_y(self.yaw0 + dyaw)
        s = self.scale0 * math.exp(logs)
        return (self.v_local * s) @ R.T + (self.pos0 + np.array([dx, 0.0, dz]))

    def world_front(self, dyaw: float) -> np.ndarray:
        return self.local_front @ _rot_y(self.yaw0 + dyaw).T


# ── Energy terms ─────────────────────────────────────────────────────────

def e_silhouette(o: Obj, wv: np.ndarray, cam: Camera) -> float:
    """1 - IoU between the projected bbox and the segmentation bbox."""
    if not o.box_px:
        return 0.0
    px = cam.project(wv)
    if len(px) < 3:
        return 1.0
    ax0, ay0 = px[:, 0].min(), px[:, 1].min()
    ax1, ay1 = px[:, 0].max(), px[:, 1].max()
    bx0, by0, bx1, by1 = (float(t) for t in o.box_px)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return 1.0 - (inter / union if union > 1e-9 else 0.0)


def e_wall(o: Obj, wv: np.ndarray, room_w: float, room_d: float) -> float:
    """Wall semantics: the back face should sit flush against its wall."""
    if o.wall not in ("back", "left", "right"):
        return 0.0
    if o.wall == "back":
        gap = wv[:, 2].min() - 0.0
    elif o.wall == "left":
        gap = wv[:, 0].min() - 0.0
    else:
        gap = room_w - wv[:, 0].max()
    return float((gap - WALL_GAP) ** 2)


def e_facing(o: Obj, dyaw: float, center: np.ndarray, tgt_center: np.ndarray) -> float:
    """facing_toward: the object's front should point at its target."""
    d = tgt_center - center
    d[1] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-6:
        return 0.0
    f = o.world_front(dyaw)
    fn = np.linalg.norm(f)
    if fn < 1e-6:
        return 0.0
    return float(1.0 - np.dot(f / fn, d / n))


def _obb_xz(wv: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Oriented XZ footprint: (centre2, half-extents2, yaw)."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rt = np.array([[c, -s], [s, c]])
    p = wv[:, [0, 2]] @ Rt.T
    lo, hi = p.min(axis=0), p.max(axis=0)
    ctr_local = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    cb, sb = math.cos(yaw), math.sin(yaw)
    Rb = np.array([[cb, -sb], [sb, cb]])
    return ctr_local @ Rb.T, half, yaw


def _sat_penetration(a: tuple, b: tuple) -> float:
    """Min penetration depth between two oriented rects (0 if separated)."""
    (ca, ha, ya), (cb, hb, yb) = a, b
    best = float("inf")
    for yaw in (ya, yb):
        c, s = math.cos(yaw), math.sin(yaw)
        for axis in (np.array([c, s]), np.array([-s, c])):
            ea = ha[0] * abs(np.dot(axis, [math.cos(ya), math.sin(ya)])) + \
                 ha[1] * abs(np.dot(axis, [-math.sin(ya), math.cos(ya)]))
            eb = hb[0] * abs(np.dot(axis, [math.cos(yb), math.sin(yb)])) + \
                 hb[1] * abs(np.dot(axis, [-math.sin(yb), math.cos(yb)]))
            dist = abs(np.dot(cb - ca, axis))
            overlap = ea + eb - dist
            if overlap <= 0.0:
                return 0.0
            best = min(best, overlap)
    return float(best)


def e_room(wv: np.ndarray, room_w: float, room_d: float) -> float:
    """Keep the footprint inside the walls."""
    out = 0.0
    out += max(0.0, -wv[:, 0].min()) ** 2
    out += max(0.0, wv[:, 0].max() - room_w) ** 2
    out += max(0.0, -wv[:, 2].min()) ** 2
    out += max(0.0, wv[:, 2].max() - room_d) ** 2
    return float(out)


def e_prior(dx: float, dz: float, dyaw: float, logs: float) -> float:
    return float((dx / PRIOR_XZ_M) ** 2 + (dz / PRIOR_XZ_M) ** 2
                 + (dyaw / PRIOR_YAW_RAD) ** 2 + (logs / PRIOR_LOGS) ** 2)


_DEPTH_RANK = {"far": 0, "mid": 1, "close": 2}


def e_depth_pair(oa: Obj, ca: np.ndarray, ob: Obj, cb: np.ndarray) -> float:
    """The VLM's own depth ordering — which the cascade never reads.

    Camera sits at large +Z, so 'close' must have a LARGER z than 'mid',
    and 'mid' larger than 'far'.  Only applied within one analysis group,
    where the relative ordering is meaningful.
    """
    if not oa.group or oa.group != ob.group:
        return 0.0
    ra = _DEPTH_RANK.get(oa.depth_tag)
    rb = _DEPTH_RANK.get(ob.depth_tag)
    if ra is None or rb is None or ra == rb:
        return 0.0
    near, far = (oa, ob) if ra > rb else (ob, oa)
    z_near = ca[2] if near is oa else cb[2]
    z_far  = cb[2] if near is oa else ca[2]
    return float(max(0.0, DEPTH_MARGIN - (z_near - z_far)) ** 2)


# ── Solver ───────────────────────────────────────────────────────────────

class Solver:
    def __init__(self, objs: list[Obj], cam: Camera, room_w: float, room_d: float):
        self.objs, self.cam = objs, cam
        self.room_w, self.room_d = room_w, room_d
        self.by_idx = {o.idx: o for o in objs}
        self.n = len(objs)

    def unpack(self, x: np.ndarray, free_yaw: bool):
        """Params -> per-object (dx, dz, dyaw, logs)."""
        k = 4 if free_yaw else 3
        out = []
        for i in range(self.n):
            c = x[i * k:(i + 1) * k]
            out.append((c[0], c[1], c[2] if free_yaw else 0.0,
                        c[3] if free_yaw else c[2]))
        return out

    def energy(self, x: np.ndarray, free_yaw: bool, breakdown: bool = False):
        params = self.unpack(x, free_yaw)
        wvs, ctrs, obbs = [], [], []
        for o, (dx, dz, dyaw, logs) in zip(self.objs, params):
            wv = o.world_verts(dx, dz, dyaw, logs)
            wvs.append(wv)
            ctrs.append(np.array([wv[:, 0].mean(), 0.0, wv[:, 2].mean()]))
            obbs.append(_obb_xz(wv, o.yaw0 + dyaw))

        parts = dict.fromkeys(W, 0.0)
        for i, (o, wv) in enumerate(zip(self.objs, wvs)):
            dx, dz, dyaw, logs = params[i]
            parts["sil"]   += e_silhouette(o, wv, self.cam)
            parts["wall"]  += e_wall(o, wv, self.room_w, self.room_d)
            parts["room"]  += e_room(wv, self.room_w, self.room_d)
            parts["prior"] += e_prior(dx, dz, dyaw, logs)
            if o.facing is not None and o.facing in self.by_idx:
                j = self.objs.index(self.by_idx[o.facing])
                parts["face"] += e_facing(o, dyaw, ctrs[i], ctrs[j])

        for i in range(self.n):
            for j in range(i + 1, self.n):
                pen = _sat_penetration(obbs[i], obbs[j])
                if pen > 0.0:
                    # A chair tucked at its desk legitimately shares footprint;
                    # allow a small tolerance for declared facing pairs.
                    tol = 0.12 if (self.objs[i].facing == self.objs[j].idx or
                                   self.objs[j].facing == self.objs[i].idx) else 0.0
                    parts["coll"] += max(0.0, pen - tol) ** 2
                parts["depth"] += e_depth_pair(self.objs[i], ctrs[i],
                                               self.objs[j], ctrs[j])

        total = sum(W[k] * v for k, v in parts.items())
        if breakdown:
            return total, {k: W[k] * v for k, v in parts.items()}
        return total

    def solve(self, verbose: bool = True):
        e0, b0 = self.energy(np.zeros(self.n * 4), True, breakdown=True)
        if verbose:
            print(f"[solver] initial E = {e0:.4f}")
            print("         " + "  ".join(f"{k}={v:.3f}" for k, v in b0.items()))

        # Stage 1 — translation + scale only (yaw frozen), as VIGA does.
        x1 = np.zeros(self.n * 3)
        r1 = minimize(self.energy, x1, args=(False,), method="Powell",
                      options={"maxiter": 4000, "xtol": 1e-3, "ftol": 1e-4})
        e1 = r1.fun
        if verbose:
            print(f"[solver] stage 1 (dx,dz,scale)  E = {e0:.4f} → {e1:.4f}")

        # Seed stage 2 from stage 1, then unlock yaw.
        x2 = np.zeros(self.n * 4)
        for i in range(self.n):
            x2[i * 4 + 0] = r1.x[i * 3 + 0]
            x2[i * 4 + 1] = r1.x[i * 3 + 1]
            x2[i * 4 + 3] = r1.x[i * 3 + 2]
        r2 = minimize(self.energy, x2, args=(True,), method="Powell",
                      options={"maxiter": 6000, "xtol": 1e-3, "ftol": 1e-5})
        e2, b2 = self.energy(r2.x, True, breakdown=True)
        if verbose:
            print(f"[solver] stage 2 (+yaw)         E = {e1:.4f} → {e2:.4f}")
            print("         " + "  ".join(f"{k}={v:.3f}" for k, v in b2.items()))
            print(f"[solver] converged={r2.success}  ΔE={e0 - e2:+.4f}")
        return self.unpack(r2.x, True), e0, e2


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--out", default="furniture_placements_solved.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scene = Path(args.scene_dir)
    furn  = scene / "furniture"
    placements = json.loads((furn / "furniture_placements.json").read_text())
    analysis   = json.loads((furn / "placement_analysis.json").read_text())
    cam_path   = scene / ("camera_vggt.json" if (scene / "camera_vggt.json").exists()
                          else "camera.json")
    cam = Camera(json.loads(cam_path.read_text()))

    by_index = {e["index"]: e for e in analysis["placement_order"]}

    room_w = room_d = None
    try:
        walls = trimesh.load(str(scene / "walls.obj"), force="mesh")
        lo, hi = walls.bounds
        room_w, room_d = float(hi[0] - lo[0]), float(hi[2] - lo[2])
    except Exception:
        room_w, room_d = 4.8, 4.4
    print(f"[solver] room {room_w:.2f} x {room_d:.2f} m   camera {cam_path.name}")

    objs = []
    for p in placements:
        if p.get("is_carpet") or p.get("wall_mounted"):
            continue
        if "position_m" not in p or "rotation_3x3" not in p:
            continue
        try:
            objs.append(Obj(p, by_index.get(p.get("index"), {}), scene))
        except Exception as exc:          # a missing GLB shouldn't kill the solve
            print(f"[solver] skip idx={p.get('index')}: {exc}")
    print(f"[solver] solving {len(objs)} objects, {len(objs) * 4} DoF")

    params, e0, e2 = Solver(objs, cam, room_w, room_d).solve()

    print(f"\n{'idx':>3} {'type':<14} {'Δx':>7} {'Δz':>7} {'Δyaw°':>7} {'Δscale':>7}")
    for o, (dx, dz, dyaw, logs) in zip(objs, params):
        print(f"{o.idx:>3} {o.type:<14} {dx:>7.3f} {dz:>7.3f} "
              f"{math.degrees(dyaw):>7.1f} {math.exp(logs):>7.3f}")

    if args.dry_run:
        print("\n[solver] --dry-run, nothing written")
        return

    for o, (dx, dz, dyaw, logs) in zip(objs, params):
        o.p["position_m"] = (o.pos0 + np.array([dx, 0.0, dz])).tolist()
        o.p["rotation_3x3"] = _rot_y(o.yaw0 + dyaw).tolist()
        o.p["scale"] = (o.scale0 * math.exp(logs)).tolist()
        o.p["_solver"] = {"dx": dx, "dz": dz, "dyaw_deg": math.degrees(dyaw),
                          "scale_mult": math.exp(logs)}
    out = furn / args.out
    out.write_text(json.dumps(placements, indent=2))
    print(f"\n[solver] E {e0:.4f} → {e2:.4f}   wrote {out}")


if __name__ == "__main__":
    main()
