"""Per-dataset / per-method image loaders.

Each loader takes a root directory and returns ``{key: image_path}``, where
the key is the matching identifier shared across loaders (typically the
input image stem, e.g. ``bedroom00``, ``livingroom1``).

To add a new method:
  1. Write a function ``load_xxx(root) -> dict[str, str]``.
  2. Register it in ``LOADERS`` below.
"""
import os

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".tiff")


def _norm_key(stem: str) -> str:
    """Normalize matching keys so loaders agree on naming.

    3D-RE-GEN (and likely other methods) replace ``,`` with ``_`` when
    creating per-scene directories, so do the same here. Also strip spaces.
    """
    return stem.replace(",", "_").replace(" ", "_")


def load_flat(root: str) -> dict[str, str]:
    """Flat directory of input images: ``root/{stem}.{ext}``.

    Used for ``testing_first1``: each file's stem (filename without
    extension) becomes the key. Falls back to alphabetical first match if
    the same stem appears with multiple extensions.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for f in sorted(os.listdir(root)):
        if f.startswith("."):
            continue
        if not f.lower().endswith(IMG_EXTS):
            continue
        stem = _norm_key(os.path.splitext(f)[0])
        out.setdefault(stem, os.path.join(root, f))
    return out


def load_3dregen(root: str) -> dict[str, str]:
    """3D-RE-GEN: ``root/{scene_name}/render_cam1_white_bg.png``."""
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, name)
        if not os.path.isdir(scene_dir):
            continue
        candidate = os.path.join(scene_dir, "render_cam1_white_bg.png")
        if os.path.isfile(candidate):
            out[_norm_key(name)] = candidate
    return out


def load_gen3dsr(root: str) -> dict[str, str]:
    """Gen3DSR: prefer paper-quality BlenderProc render; fall back to Open3D
    PBR render if BlenderProc one isn't available for that scene.

    Files (in priority order):
        1. ``{scene}/render_bproc.png`` — BlenderProc/Cycles 128 spp (paper)
        2. ``{scene}/render_input_view.png`` — Open3D PBR (faster fallback)
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, name)
        if not os.path.isdir(scene_dir):
            continue
        for fname in ("render_bproc.png", "render_input_view.png"):
            candidate = os.path.join(scene_dir, fname)
            if os.path.isfile(candidate):
                out[_norm_key(name)] = candidate
                break
    return out


def load_harmony(root: str) -> dict[str, str]:
    """HARMONY: prefer the latest pipeline-stage render available.

    ``root`` may be a single directory or multiple comma-separated roots.
    When multiple roots are given, **earlier roots take precedence** for any
    scene name that appears in more than one. This lets the caller order
    "curated" output (e.g. ``Artifact/``) before "raw batch" output
    (e.g. ``_processed/``).

    Priority order within a scene dir (later stage wins):
        1. ``{scene}/decorations/placements/render_decorations_placed.png``
        2. ``{scene}/furniture/render_furniture_placed.png`` (fallback for
           scenes that haven't finished the decoration pass yet)

    NOTE: the newer lit-Blender render (``lightings/render_blender.png``) is
    NOT picked up here — use the dedicated ``harmony_blender`` loader for that,
    so this loader's behavior (and the documented reference numbers) stays
    stable.
    """
    # Prefer the wall-textured decoration render; fall back through the wall-tex
    # furniture stage, then plain variants.
    candidates_in_order = [
        ("decorations", "placements", "render_decorations_placed_with_new_wall_texture.png"),
        ("furniture", "render_furniture_placed_with_new_wall_texture.png"),
        ("decorations", "placements", "render_decorations_placed.png"),
        ("furniture", "render_furniture_placed.png"),
    ]
    roots = [r.strip() for r in root.split(",") if r.strip()]
    out: dict[str, str] = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            scene_dir = os.path.join(r, name)
            if not os.path.isdir(scene_dir):
                continue
            key = _norm_key(name)
            if key in out:
                continue  # earlier roots take precedence
            for parts in candidates_in_order:
                path = os.path.join(scene_dir, *parts)
                if os.path.isfile(path):
                    out[key] = path
                    break
    return out


def load_harmony_blender(root: str) -> dict[str, str]:
    """Harmony newest render: ``{root}/{scene}/lightings/render_blender.png``.

    The final lit-Blender pass. Kept as a SEPARATE loader from ``load_harmony``
    so the original decorations/furniture reference behavior is never altered.

    ``root`` may be comma-separated multi-root (earlier roots win), matching
    ``load_harmony``'s convention.
    """
    roots = [r.strip() for r in root.split(",") if r.strip()]
    out: dict[str, str] = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            scene_dir = os.path.join(r, name)
            if not os.path.isdir(scene_dir):
                continue
            key = _norm_key(name)
            if key in out:
                continue  # earlier roots take precedence
            path = os.path.join(scene_dir, "lightings", "render_blender.png")
            if os.path.isfile(path):
                out[key] = path
    return out


def load_harmony_ambient(root: str) -> dict[str, str]:
    """Harmony ambient-lit render: ``{root}/{scene}/lightings/render_blender_ambient.png``.

    Same lightings stage as ``load_harmony_blender`` but the ambient-lighting
    variant. Separate loader so each render condition is selectable on its own.
    ``root`` may be comma-separated multi-root (earlier roots win).
    """
    roots = [r.strip() for r in root.split(",") if r.strip()]
    out: dict[str, str] = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            scene_dir = os.path.join(r, name)
            if not os.path.isdir(scene_dir):
                continue
            key = _norm_key(name)
            if key in out:
                continue  # earlier roots take precedence
            path = os.path.join(scene_dir, "lightings", "render_blender_ambient.png")
            if os.path.isfile(path):
                out[key] = path
    return out


def load_harmony_unlit(root: str) -> dict[str, str]:
    """Harmony UNLIT render — latest pre-lighting stage available per scene.

    Same scene dirs as ``load_harmony_blender`` (e.g. _Demo_June/_finished),
    but picks the newest *unlit* stage instead of lightings/render_blender.png:

        1. ``{scene}/ceiling/render_ceiling_placed.png``          (ceiling stage)
        2. ``{scene}/decorations/placements/render_decorations_placed.png``
        3. ``{scene}/furniture/render_furniture_placed.png``      (earliest)

    Kept separate from ``load_harmony`` so nothing about the reference changes.
    ``root`` may be comma-separated multi-root (earlier roots win).
    """
    candidates_in_order = [
        ("ceiling", "render_ceiling_placed.png"),
        ("decorations", "placements", "render_decorations_placed.png"),
        ("furniture", "render_furniture_placed.png"),
    ]
    roots = [r.strip() for r in root.split(",") if r.strip()]
    out: dict[str, str] = {}
    for r in roots:
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            scene_dir = os.path.join(r, name)
            if not os.path.isdir(scene_dir):
                continue
            key = _norm_key(name)
            if key in out:
                continue  # earlier roots take precedence
            for parts in candidates_in_order:
                path = os.path.join(scene_dir, *parts)
                if os.path.isfile(path):
                    out[key] = path
                    break
    return out


def load_sam3d(root: str) -> dict[str, str]:
    """SAM-3D scene render: ``{root}/**/render_meshscene.png``.

    Keyed by the immediate parent directory name (the scene id). Uses os.walk
    so it works whether the render sits directly under ``{root}/{scene}/`` or
    is nested deeper. If the same scene id appears more than once, the first
    (shallowest, then alphabetical) match wins.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        if "render_meshscene.png" in filenames:
            key = _norm_key(os.path.basename(dirpath))
            out.setdefault(key, os.path.join(dirpath, "render_meshscene.png"))
    return out


def _load_by_filename(root: str, filename: str) -> dict[str, str]:
    """Walk ``root`` and map each scene dir (immediate parent) -> ``filename``.

    Shared by the SAM-3D variant loaders, which differ only in which render
    PNG they pick from ``outputs/SAM3D/{scene}/``. First (shallowest, then
    alphabetical) match wins on duplicate scene ids.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        if filename in filenames:
            out.setdefault(_norm_key(os.path.basename(dirpath)),
                           os.path.join(dirpath, filename))
    return out


def load_sam3d_texbg(root: str) -> dict[str, str]:
    """SAM-3D WITH-background render: ``{root}/{scene}/render_over_ref_texture.png``."""
    return _load_by_filename(root, "render_over_ref_texture.png")


def load_sam3d_rawnobg(root: str) -> dict[str, str]:
    """SAM-3D NO-background render: ``{root}/{scene}/render_perspective_raw.png``."""
    return _load_by_filename(root, "render_perspective_raw.png")


def load_sam3d_room(root: str) -> dict[str, str]:
    """SAM-3D room-composited render: ``{root}/{scene}/render_with_room_flat.png``.

    The cleanest SAM-3D variant — furniture composited into a flat-lit room
    (walls / window / floor), not the washed-out mesh-only passes.
    """
    return _load_by_filename(root, "render_with_room_flat.png")


def load_cast(root: str) -> dict[str, str]:
    """CAST scene render: ``{root}/{scene}/{scene}/cast_with_bg.png``.

    The renders are double-nested; keyed by the immediate parent directory
    name (the inner ``{scene}``). os.walk handles the nesting; first match
    wins on duplicate scene ids.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        if "cast_with_bg.png" in filenames:
            key = _norm_key(os.path.basename(dirpath))
            out.setdefault(key, os.path.join(dirpath, "cast_with_bg.png"))
    return out


def load_cast_room(root: str) -> dict[str, str]:
    """CAST room-composited render: ``{root}/{scene}/{scene}/scene_render_room_flat.png``.

    Furniture composited into a flat-lit room. Unlike ``cast_with_bg.png`` this
    variant exists for all scenes (incl. the front3d rgb_* set), so it's what to
    use for cross-dataset pixel comparison. Same double-nested layout.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        if "scene_render_room_flat.png" in filenames:
            key = _norm_key(os.path.basename(dirpath))
            out.setdefault(key, os.path.join(dirpath, "scene_render_room_flat.png"))
    return out


def load_cast_reproj(root: str) -> dict[str, str]:
    """CAST local re-run render: ``{root}/{scene}/{scene}/reprojection_render.png``.

    The reconstructed CAST scene reprojected into the input camera (white bg,
    furniture only — no room composite). This is what the local pipeline emits;
    comparable to the raw-furniture renders (SAM3D-nobg), not the room-composited
    ``scene_render_room_flat.png``.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        if "reprojection_render.png" in filenames:
            key = _norm_key(os.path.basename(dirpath))
            out.setdefault(key, os.path.join(dirpath, "reprojection_render.png"))
    return out


# Set $ASTRA_ATTRIB to the benchmark's attribution.csv (rgb_ id -> scene name).
_ASTRA_ATTRIB_DEFAULT = os.environ.get("ASTRA_ATTRIB", "")
_astra_name2rgb_cache = None


def _astra_name2rgb() -> dict[str, str]:
    """Map astra semantic name (bedroom_01) -> front3d rgb_ id via attribution.csv.

    The easy split's 5th column (mislabelled 'pexels_id') holds the rgb_ id.
    Set $ASTRA_ATTRIB to override the csv path. Cached after first read.
    """
    global _astra_name2rgb_cache
    if _astra_name2rgb_cache is None:
        import csv
        attrib = os.environ.get("ASTRA_ATTRIB", _ASTRA_ATTRIB_DEFAULT)
        m: dict[str, str] = {}
        if os.path.isfile(attrib):
            for r in csv.DictReader(open(attrib)):
                if r.get("difficulty") == "easy":
                    m[r["name"]] = r["pexels_id"]
        _astra_name2rgb_cache = m
    return _astra_name2rgb_cache


def load_harmony_astra(root: str) -> dict[str, str]:
    """Astra Harmony release lit render: ``{root}/easy/{semantic}/render.png``.

    Scenes are semantically named; keyed by the front3d rgb_ id (via
    attribution.csv) so they line up with the front3d ref photos and GT.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    n2r = _astra_name2rgb()
    base = os.path.join(root, "easy")
    if not os.path.isdir(base):
        base = root
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name, "render.png")
        if os.path.isfile(p):
            out.setdefault(_norm_key(n2r.get(name, name)), p)
    return out


def load_harmony_astra_all(root: str) -> dict[str, str]:
    """Astra Harmony bundle with several difficulty folders:
    ``{root}/{easy|medium|complicated}/{semantic}/render.png``.
    Keys are the semantic names (FRONT3D ones mapped to rgb_ ids via attribution.csv),
    so they line up with the harmony300 canonical names used by ``load_viga`` on
    ``VIGA/results/harmony300`` and with a flat ref dir keyed by those names.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    n2r = _astra_name2rgb()
    for split in sorted(os.listdir(root)):
        base = os.path.join(root, split)
        if not os.path.isdir(base) or split.startswith("_"):
            continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name, "render.png")
            if os.path.isfile(p):
                out.setdefault(_norm_key(n2r.get(name, name)), p)
    return out


def load_harmony_input(root: str) -> dict[str, str]:
    """Input photo stored inside each Harmony scene dir: ``{root}/{scene}/{scene}.{ext}``.

    Used as the reference (ground-truth photo) when the scenes aren't in a
    flat testing_* dir — e.g. the ablation variants under _ablations, whose
    scene dirs each carry their own input image (``elegant/elegant.jpg`` …).
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, name)
        if not os.path.isdir(scene_dir):
            continue
        key = _norm_key(name)
        for ext in IMG_EXTS:
            cand = os.path.join(scene_dir, name + ext)
            if os.path.isfile(cand):
                out[key] = cand
                break
    return out


def load_scenegen_hero(root: str) -> dict[str, str]:
    """SceneGen hero render: ``root/{stem}.png``."""
    return load_flat(root)


def load_gen3dsr_gt(root: str) -> dict[str, str]:
    """Gen3DSR FRONT3D GT renders: ``root/gt_{scene_id}.png``.

    Strips the ``gt_`` prefix to expose the bare scene id as the key.
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for f in sorted(os.listdir(root)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(f)[0]
        if stem.startswith("gt_"):
            stem = stem[len("gt_"):]
        out[_norm_key(stem)] = os.path.join(root, f)
    return out


def load_viga(root: str) -> dict[str, str]:
    """VIGA static-scene runs: ``root/<task>/renders/<N>/Camera*.png``.

    Each task accumulates renders/<round>/ as the dual agent iterates.
    The highest-numbered subdir holds the final scene; we pick its
    ``Camera*.png`` (the camera name varies — sometimes ``Camera.png``,
    sometimes ``Camera_<viewname>.png``). Tasks without any final render
    are silently skipped.

    ``root`` typically points at one timestamped run, e.g.::

        VIGA/output/static_scene/20260615_065518     # prompt-setting=init
        VIGA/output/static_scene/20260614_070551     # prompt-setting=none
    """
    out: dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for task in sorted(os.listdir(root)):
        task_dir = os.path.join(root, task)
        renders_dir = os.path.join(task_dir, "renders")
        if not os.path.isdir(renders_dir):
            continue
        # Iteration subdirs are numeric strings — pick the highest
        iter_dirs = [d for d in os.listdir(renders_dir)
                     if d.isdigit() and os.path.isdir(os.path.join(renders_dir, d))]
        if not iter_dirs:
            continue
        last = max(iter_dirs, key=int)
        last_dir = os.path.join(renders_dir, last)
        # The agent renames cameras freely between rounds, so we accept any
        # PNG whose name contains the word "camera" (case-insensitive). The
        # only other file in the iteration dir is state.blend.
        cam_pngs = sorted(
            f for f in os.listdir(last_dir)
            if f.lower().endswith(".png") and "camera" in f.lower()
        )
        if not cam_pngs:
            continue
        out[_norm_key(task)] = os.path.join(last_dir, cam_pngs[0])
    return out


LOADERS = {
    "flat": load_flat,
    "3dregen": load_3dregen,
    "gen3dsr": load_gen3dsr,
    "harmony": load_harmony,
    "harmony_blender": load_harmony_blender,
    "harmony_ambient": load_harmony_ambient,
    "harmony_unlit": load_harmony_unlit,
    "harmony_input": load_harmony_input,
    "sam3d": load_sam3d,
    "sam3d_texbg": load_sam3d_texbg,
    "sam3d_rawnobg": load_sam3d_rawnobg,
    "sam3d_room": load_sam3d_room,
    "cast": load_cast,
    "cast_room": load_cast_room,
    "cast_reproj": load_cast_reproj,
    "harmony_astra": load_harmony_astra,
    "harmony_astra_all": load_harmony_astra_all,
    "viga": load_viga,
    "scenegen": load_scenegen_hero,
    "gen3dsr_gt": load_gen3dsr_gt,
}


def get_loader(name: str):
    if name not in LOADERS:
        raise ValueError(
            f"Unknown loader '{name}'. Available: {sorted(LOADERS)}"
        )
    return LOADERS[name]
