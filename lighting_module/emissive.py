"""Edit GLB materials so light-source meshes physically emit light.

For lamps and ceiling fixtures we want only the LIT region (lampshade or
glowing bulb) to be emissive — not the metal pole. We segment the lit region
in the inpaint PNG by brightness threshold, then split the mesh into:

  base_<name>      — non-emissive faces, original material
  emissive_<name>  — faces whose UV centroid falls inside the lit-region mask

The two parts are saved together as `lit_<name>.glb` in
`<output>/lightings/objects/`. Original GLBs in `decorations/objects/`
and `wall_mounted/objects/` are never modified.

Windows are handled differently: their plane mesh is wholly emissive (the
sky is the surface that glows), so we just bake an emissive copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _emissive_color_uint8(rgb01: list[float], intensity: float) -> np.ndarray:
    """Convert (linear-ish) [r,g,b] in 0..1 + intensity → 8-bit RGBA emissive
    base colour. Trimesh treats emissiveFactor through visual.material in newer
    versions; for max viewer compatibility we ALSO write a saturated base
    colour, since a self-lit material reads acceptably in the rasterizer."""
    r, g, b = (max(0.0, float(c)) for c in rgb01)
    boost = max(0.5, min(2.5, float(intensity)))
    rgb = np.array([r, g, b], dtype=np.float32) * boost
    rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    return np.array([*rgb.astype(np.uint8), 255], dtype=np.uint8)


def _lit_mask(inpaint_path: Path, *, percentile: float = 92.0) -> np.ndarray | None:
    """Return a (H, W) bool mask of the brightest connected region. None on failure."""
    try:
        img = np.array(Image.open(str(inpaint_path)).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None

    lum = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    thresh = np.percentile(lum, percentile)
    # A safety floor: if the whole image is dim, don't pick a "bright" region.
    if thresh < 120:
        return None
    mask = lum >= thresh
    if mask.sum() < 50:
        return None
    return mask


def _face_uv_centroids(mesh) -> np.ndarray | None:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None
    uv_arr = np.asarray(uv)
    if uv_arr.shape[0] != len(mesh.vertices):
        return None
    fuv = uv_arr[mesh.faces]                  # (F, 3, 2)
    return fuv.mean(axis=1)                   # (F, 2)


def _emissive_face_indices(mesh, lit_mask: np.ndarray) -> np.ndarray:
    """Return face indices whose UV centroid lies inside `lit_mask`."""
    fc = _face_uv_centroids(mesh)
    if fc is None:
        return np.array([], dtype=np.int64)
    h, w = lit_mask.shape[:2]
    u_px = np.clip((fc[:, 0] * (w - 1)).astype(int), 0, w - 1)
    v_px = np.clip(((1.0 - fc[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    return np.where(lit_mask[v_px, u_px])[0]


def _make_emissive_visual(trimesh_mod, color_rgba: np.ndarray):
    """Build a PBR material with a self-lit base + emissiveFactor."""
    try:
        from trimesh.visual.material import PBRMaterial
        mat = PBRMaterial(
            name="emissive",
            baseColorFactor=color_rgba.tolist(),
            emissiveFactor=color_rgba[:3].astype(float).tolist(),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        return trimesh_mod.visual.TextureVisuals(material=mat)
    except Exception:
        return None


def _split_and_save(trimesh_mod, mesh, em_face_idx: np.ndarray,
                    color_rgba: np.ndarray, out_path: Path) -> bool:
    base_idx = np.setdiff1d(np.arange(len(mesh.faces)), em_face_idx, assume_unique=False)
    if len(em_face_idx) == 0 or len(base_idx) == 0:
        return False

    base = mesh.submesh([base_idx], append=True)
    emit = mesh.submesh([em_face_idx], append=True)

    # Apply emissive material — works for both PBR-aware viewers and the
    # rasterizer (which falls back to vertex colours).
    em_visual = _make_emissive_visual(trimesh_mod, color_rgba)
    if em_visual is not None:
        emit.visual = em_visual
    try:
        emit.visual.vertex_colors = np.tile(color_rgba, (len(emit.vertices), 1))
    except Exception:
        pass

    scene = trimesh_mod.Scene()
    scene.add_geometry(base, geom_name="base")
    scene.add_geometry(emit, geom_name="emissive")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))
    return True


def _bake_full_emissive(trimesh_mod, mesh, color_rgba: np.ndarray,
                        out_path: Path) -> bool:
    em = mesh.copy()
    em_visual = _make_emissive_visual(trimesh_mod, color_rgba)
    if em_visual is not None:
        em.visual = em_visual
    try:
        em.visual.vertex_colors = np.tile(color_rgba, (len(em.vertices), 1))
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh_mod.Scene([em]).export(str(out_path))
    return True


def edit_one(source: dict, light_params: dict, lit_objects_dir: Path) -> dict | None:
    """Apply emissive material to a single light source's GLB.

    Returns a dict describing the lit GLB (path, kind, fallback_used) or None
    if the GLB couldn't be loaded.
    """
    import trimesh

    glb_in = source.get("glb_path")
    if not glb_in:
        return None
    glb_in = Path(glb_in)
    if not glb_in.exists():
        print(f"[lighting/emissive] missing GLB {glb_in}")
        return None

    color01 = light_params.get("color", [1.0, 0.85, 0.6])
    intensity = float(light_params.get("intensity", 1.0))
    on = bool(light_params.get("on", light_params.get("daylight_on", True)))
    # A ceiling LIGHT fixture (chandelier/pendant/ceiling lamp) is a luminaire by
    # definition — give its shade/bulbs a warm emission floor even when the VLM
    # judged it "off"/not glowing, so it reads as a light in the render instead
    # of dark glass.
    if source.get("kind") == "ceiling_fixture" and (not on or intensity <= 1e-3):
        on = True
        intensity = max(intensity, 0.7)
        if not light_params.get("color"):
            color01 = [1.0, 0.85, 0.6]
    if not on or intensity <= 1e-3:
        return None

    color_rgba = _emissive_color_uint8(color01, intensity)

    try:
        loaded = trimesh.load(str(glb_in), force="mesh")
        mesh = loaded if isinstance(loaded, trimesh.Trimesh) else trimesh.util.concatenate(
            list(loaded.dump()) if isinstance(loaded, trimesh.Scene) else [loaded])
    except Exception as e:
        print(f"[lighting/emissive] load failed for {glb_in.name}: {e}")
        return None

    out_path = lit_objects_dir / f"lit_{glb_in.stem}.glb"

    # Windows: emit from the entire plane.
    if source["kind"] == "window":
        ok = _bake_full_emissive(trimesh, mesh, color_rgba, out_path)
        return {"id": source["id"], "kind": source["kind"],
                "glb_in": str(glb_in), "glb_out": str(out_path),
                "mode": "full" if ok else "failed"}

    # Lamps / ceiling fixtures: split by inpaint brightness mask.
    inpaint = source.get("inpaint")
    em_face_idx = np.array([], dtype=np.int64)
    if inpaint:
        mask = _lit_mask(Path(inpaint))
        if mask is not None:
            em_face_idx = _emissive_face_indices(mesh, mask)

    if len(em_face_idx) >= 8:
        ok = _split_and_save(trimesh, mesh, em_face_idx, color_rgba, out_path)
        mode = "split" if ok else "split_failed"
    else:
        # Fallback: top 30 % of the mesh (by Y) — a reasonable shade proxy.
        y = mesh.vertices[:, 1]
        cutoff = np.percentile(y, 70)
        v_em = y >= cutoff
        f_em = np.where(v_em[mesh.faces].any(axis=1))[0]
        if len(f_em) >= 8:
            ok = _split_and_save(trimesh, mesh, f_em, color_rgba, out_path)
            mode = "top_third" if ok else "top_third_failed"
        else:
            ok = _bake_full_emissive(trimesh, mesh, color_rgba, out_path)
            mode = "full_fallback" if ok else "failed"

    return {"id": source["id"], "kind": source["kind"],
            "glb_in": str(glb_in), "glb_out": str(out_path),
            "mode": mode}


def edit_all(sources: list[dict], lights: dict, lit_objects_dir: Path) -> list[dict]:
    """Edit emissive materials for every active light source. Returns the list
    of records describing what was written to disk."""
    by_id_pl = {p["id"]: p for p in lights.get("point_lights", [])}
    by_id_w  = {w["id"]: w for w in lights.get("windows", [])}

    edits: list[dict] = []
    for s in sources:
        sid = s.get("id")
        if not sid:
            continue
        if s["kind"] in ("lamp", "ceiling_fixture"):
            params = by_id_pl.get(sid)
        elif s["kind"] == "window":
            params = by_id_w.get(sid)
        else:
            params = None
        if params is None:
            continue
        rec = edit_one(s, params, lit_objects_dir)
        if rec is not None:
            edits.append(rec)
    return edits
