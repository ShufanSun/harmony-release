"""Depth-guided wall-extent refinement.

After single-corner camera calibration, a wall's *length* (where it ends and the
perpendicular wall begins) is under-constrained -- it follows from automatic
scaling and can land too long or too short.  This module uses the VGGT metric
depth to check, per pixel, whether the estimated wall plane agrees with the
measured surface.  Where an empty wall region's measured depth is consistently
closer/farther than the box predicts, the wall is mis-placed and its plane is
re-fit from the back-projected measured points (classified by their own normal,
so a "back wall that actually extends further right" is handled correctly rather
than naively shoving the side wall outward).

Inputs per scene dir:  camera_vggt.json, walls.obj, vggt/depth_0.npy, rgb_*.jpeg
Outputs:  walls_refined.obj + visualizations (_depth_residual / _wall_classify /
_footprint_topdown).
"""
import os, glob, json
import numpy as np
import cv2


def _load_camera(scene_dir):
    cam = json.loads(open(os.path.join(scene_dir, "camera_vggt.json")).read())
    W, H = int(cam["width_px"]), int(cam["height_px"])
    pos = np.array(cam["position_m"], float)
    look = np.array(cam["look_at_m"], float)
    fx = W / (2 * np.tan(np.radians(cam["hfov_deg"] / 2)))
    fy = H / (2 * np.tan(np.radians(cam.get("vfov_deg", cam["hfov_deg"]) / 2)))
    fwd = look - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return cam, W, H, pos, fwd, right, up, fx, fy


def _ray_dirs(W, H, pos, fwd, right, up, fx, fy):
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    return (((us + 0.5 - W / 2) / fx)[..., None] * right
            + (-(vs + 0.5 - H / 2) / fy)[..., None] * up
            + fwd[None, None, :])


def _box_bounds(scene_dir):
    verts = np.array([[float(x) for x in l.split()[1:4]]
                      for l in open(os.path.join(scene_dir, "walls.obj"))
                      if l.startswith("v ")])
    return verts.min(0), verts.max(0)


def _predict_depth(dirs, pos, fwd, lo, hi):
    """Nearest bounded box-plane hit per ray -> predicted depth (along fwd) + wall id."""
    H, W = dirs.shape[:2]
    planes = [(0, lo[0], (1, 2)), (0, hi[0], (1, 2)),
              (2, lo[2], (0, 1)), (2, hi[2], (0, 1)),
              (1, lo[1], (0, 2)), (1, hi[1], (0, 2))]
    best_t = np.full((H, W), np.inf); wid = np.full((H, W), -1, int)
    for k, (axis, val, (a, b)) in enumerate(planes):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - pos[axis]) / dirs[..., axis]
        ha = pos[a] + t * dirs[..., a]; hb = pos[b] + t * dirs[..., b]
        ins = ((t > 1e-4) & np.isfinite(t)
               & (ha >= lo[a] - 1e-3) & (ha <= hi[a] + 1e-3)
               & (hb >= lo[b] - 1e-3) & (hb <= hi[b] + 1e-3) & (t < best_t))
        best_t = np.where(ins, t, best_t); wid = np.where(ins, k, wid)
    Dpred = best_t * (dirs * fwd).sum(2)
    return Dpred, wid  # wid: 0 x_lo,1 x_hi,2 z_lo,3 z_hi,4 floor,5 ceil


def _align_depth(dv, Dpred, valid):
    """Robust affine fit measured(vggt) -> predicted(room m), rejecting occluders."""
    inl = valid.copy(); a, b = 1.0, 0.0
    for _ in range(6):
        if inl.sum() < 100:
            break
        A = np.vstack([dv[inl], np.ones(inl.sum())]).T
        (a, b), *_ = np.linalg.lstsq(A, Dpred[inl], rcond=None)
        r = (a * dv + b) - Dpred
        sd = np.std(r[inl]) + 1e-6
        inl = valid & (np.abs(r) < 1.5 * sd)
    return a * dv + b, a, b, inl


def _point_normals(P, valid):
    """Per-pixel normal from the 3D point map (room frame). Returns unit normals + ok mask."""
    H, W = P.shape[:2]
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=2, keepdims=True)
    ok = valid & (ln[..., 0] > 1e-6)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-6)
    # consistent step: also flag where neighbours valid
    nb = valid.copy()
    nb[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    nb[1:-1, :] &= valid[2:, :] & valid[:-2, :]
    return n, ok & nb


def refine_walls_with_depth(scene_dir, shift_thresh=0.30, pct=2.0,
                            normal_tol=0.6, save_vis=True):
    cam, W, H, pos, fwd, right, up, fx, fy = _load_camera(scene_dir)
    dirs = _ray_dirs(W, H, pos, fwd, right, up, fx, fy)
    lo, hi = _box_bounds(scene_dir)
    Dpred, wid = _predict_depth(dirs, pos, fwd, lo, hi)

    dv = np.load(os.path.join(scene_dir, "vggt", "depth_0.npy")).astype(float)
    valid = np.isfinite(Dpred) & (Dpred > 0.1) & (dv > 0.05)
    Dmeas, a, b, inl = _align_depth(dv, Dpred, valid)
    valid &= np.isfinite(Dmeas) & (Dmeas > 0.1) & (Dmeas < 30)
    resid = Dmeas - Dpred

    # back-project measured depth into room frame
    dirfwd = (dirs * fwd).sum(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = Dmeas / dirfwd
    P = pos[None, None, :] + s[..., None] * dirs

    n, nok = _point_normals(P, valid)
    # classify by dominant room axis of the normal
    ax = np.argmax(np.abs(n), axis=2)          # 0=x,1=y,2=z
    mag = np.max(np.abs(n), axis=2)
    cls = np.full((H, W), -1, int)             # -1 none, 0 sidewall(x), 1 floor/ceil(y), 2 backwall(z)
    cls[nok & (mag > normal_tol)] = ax[nok & (mag > normal_tol)]

    # --- re-fit footprint from measured WALL/FLOOR points (interior furniture
    #     can't extend the outer hull, so robust percentiles give room extent) ---
    # A wall's extent must come from the VERTICAL surfaces (cls 0=x-wall, 2=z-wall)
    # at their deepest reach -- the floor is nearer/noisier and would bias the
    # extent.  We bound each side by the deepest robust percentile of wall points.
    floorlike = valid & (cls == 1) & (P[..., 1] < (lo[1] + hi[1]) * 0.5)   # for viz only
    walllike = valid & np.isin(cls, [0, 2])
    foot = walllike
    Px, Pz = P[..., 0][walllike], P[..., 2][walllike]
    new_lo = lo.copy(); new_hi = hi.copy()

    def wall_support(axis, plane, tol=0.3, need=1500):
        """# measured points whose normal matches this wall axis, lying near `plane`."""
        m = valid & (cls == axis) & (np.abs(P[..., axis] - plane) < tol)
        return int(m.sum()) >= need

    # box ray ids: 0 x_lo,1 x_hi,2 z_lo,3 z_hi.  A wall is "in view" only if the
    # box casts enough rays onto it (the camera faces it) -- a wall behind the
    # camera (e.g. the front wall) gets ~0 rays and must not be shrunk.
    def in_view(wall_id, need=1500):
        return int((wid == wall_id).sum()) >= need

    if Px.size > 2000:
        cand_lo_x = np.percentile(Px, pct); cand_hi_x = np.percentile(Px, 100 - pct)
        cand_lo_z = np.percentile(Pz, pct); cand_hi_z = np.percentile(Pz, 100 - pct)
        # EXPAND freely (measured surface lies beyond the box -> room is bigger);
        # SHRINK only for an in-view wall with normal support for a closer plane.
        for axis, lo_id, hi_id, cand_lo_v, cand_hi_v in [
                (0, 0, 1, cand_lo_x, cand_hi_x), (2, 2, 3, cand_lo_z, cand_hi_z)]:
            if cand_lo_v < lo[axis] - shift_thresh:                       # expand low
                new_lo[axis] = cand_lo_v
            elif (cand_lo_v > lo[axis] + shift_thresh
                  and in_view(lo_id) and wall_support(axis, cand_lo_v)):
                new_lo[axis] = cand_lo_v                                  # closer wall (shrink)
            if cand_hi_v > hi[axis] + shift_thresh:                       # expand high
                new_hi[axis] = cand_hi_v
            elif (cand_hi_v < hi[axis] - shift_thresh
                  and in_view(hi_id) and wall_support(axis, cand_hi_v)):
                new_hi[axis] = cand_hi_v                                  # closer wall (shrink)
    new_lo[1], new_hi[1] = lo[1], hi[1]   # keep height as calibrated
    report = {
        "scale": float(a), "shift": float(b), "inlier_frac": float(inl.mean()),
        "resid_sd": float(np.std(resid[inl])) if inl.any() else None,
        "box_old": {"lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()},
        "box_new": {"lo": new_lo.round(3).tolist(), "hi": new_hi.round(3).tolist()},
        "delta": {"x_lo": float(new_lo[0]-lo[0]), "x_hi": float(new_hi[0]-hi[0]),
                  "z_lo": float(new_lo[2]-lo[2]), "z_hi": float(new_hi[2]-hi[2])},
    }
    # per-assigned-wall median residual over empty (non-occluded) upper region
    upper = np.zeros((H, W), bool); upper[:int(0.5*H)] = True
    names = ["x_lo", "x_hi", "z_lo", "z_hi", "floor", "ceil"]
    perwall = {}
    for k in range(4):
        m = valid & (wid == k) & upper & (resid > -0.3)   # drop obvious occluders
        if m.sum() > 300:
            perwall[names[k]] = {"px": int(m.sum()),
                                 "median_resid_m": float(np.median(resid[m]))}
    report["per_wall_empty"] = perwall

    # only keep deltas that exceed threshold; otherwise snap back to original
    for i, nm in [(0, "x_lo"), (2, "z_lo")]:
        if abs(new_lo[i] - lo[i]) < shift_thresh:
            new_lo[i] = lo[i]
    for i, nm in [(0, "x_hi"), (2, "z_hi")]:
        if abs(new_hi[i] - hi[i]) < shift_thresh:
            new_hi[i] = hi[i]
    report["box_new_thresh"] = {"lo": new_lo.round(3).tolist(), "hi": new_hi.round(3).tolist()}

    template = os.path.join(scene_dir, "walls.obj")
    _write_box_from_template(os.path.join(scene_dir, "walls_refined.obj"),
                             template, lo, hi, new_lo, new_hi)

    if save_vis:
        _save_vis(scene_dir, W, H, resid, valid, cls, P, foot | floorlike,
                  lo, hi, new_lo, new_hi, pos, fwd, right)
    return report


def _write_box_from_template(path, template_obj, old_lo, old_hi, new_lo, new_hi, tol=1e-3):
    """Rewrite the box by remapping the *original* obj's vertices to the new
    bounds.  Preserves vertex order, face list and winding (inward normals) so
    the interior render stays lit -- a freshly-authored box can flip normals."""
    out = []
    for l in open(template_obj):
        if l.startswith("v "):
            p = [float(v) for v in l.split()[1:4]]
            for ax in range(3):
                if abs(p[ax] - old_lo[ax]) < tol:   p[ax] = new_lo[ax]
                elif abs(p[ax] - old_hi[ax]) < tol: p[ax] = new_hi[ax]
            out.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            out.append(l)
    with open(path, "w") as f:
        f.writelines(out)


def _save_vis(scene_dir, W, H, resid, valid, cls, P, foot, lo, hi, nlo, nhi, pos, fwd, right):
    photo_p = (glob.glob(os.path.join(scene_dir, "rgb_*.jpeg"))
               + glob.glob(os.path.join(scene_dir, "rgb_*.png")))
    photo = cv2.resize(cv2.imread(photo_p[0]), (W, H)) if photo_p else np.zeros((H, W, 3), np.uint8)
    # residual overlay
    ov = photo.copy()
    ov[valid & (resid < -0.5)] = (0, 0, 255)   # measured closer
    ov[valid & (resid > 0.5)] = (255, 0, 0)    # measured farther
    vis = cv2.addWeighted(photo, 0.45, ov, 0.55, 0)
    cv2.imwrite(os.path.join(scene_dir, "_depth_residual.png"), vis)
    # wall classification
    col = {0: (0, 165, 255), 1: (0, 200, 0), 2: (255, 80, 0)}  # x=orange z=blue floor=green
    cm = photo.copy()
    for k, c in col.items():
        cm[valid & (cls == k)] = c
    cv2.imwrite(os.path.join(scene_dir, "_wall_classify.png"),
                cv2.addWeighted(photo, 0.4, cm, 0.6, 0))
    # top-down footprint: measured x,z points + old vs new box rectangle
    Px, Pz = P[..., 0][foot], P[..., 2][foot]
    allx = np.concatenate([Px, [lo[0], hi[0], nlo[0], nhi[0], pos[0]]])
    allz = np.concatenate([Pz, [lo[2], hi[2], nlo[2], nhi[2], pos[2]]])
    x0, x1, z0, z1 = allx.min()-0.3, allx.max()+0.3, allz.min()-0.3, allz.max()+0.3
    S = 900
    sx = lambda x: int((x - x0) / (x1 - x0) * (S - 1))
    sz = lambda z: int((S - 1) - (z - z0) / (z1 - z0) * (S - 1))
    canvas = np.full((S, S, 3), 255, np.uint8)
    idx = np.linspace(0, Px.size - 1, min(Px.size, 40000)).astype(int)
    for x, z in zip(Px[idx], Pz[idx]):
        cv2.circle(canvas, (sx(x), sz(z)), 1, (160, 160, 160), -1)
    def rect(L, Hh, color, th):
        cv2.rectangle(canvas, (sx(L[0]), sz(L[2])), (sx(Hh[0]), sz(Hh[2])), color, th)
    rect(lo, hi, (0, 0, 255), 2)      # original box red
    rect(nlo, nhi, (0, 160, 0), 2)    # refined box green
    cv2.circle(canvas, (sx(pos[0]), sz(pos[2])), 6, (255, 0, 0), -1)  # camera
    cv2.putText(canvas, "red=orig  green=depth-refined  blue=cam", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(scene_dir, "_footprint_topdown.png"), canvas)


if __name__ == "__main__":
    import sys
    sd = sys.argv[1] if len(sys.argv) > 1 else "outputs/front3d/rgb_003454"
    rep = refine_walls_with_depth(sd)
    print(json.dumps(rep, indent=2))
