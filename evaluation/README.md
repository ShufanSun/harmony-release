# evaluation — HARMONY300 scoring

Scoring code for the **HARMONY300** benchmark
([dataset](https://huggingface.co/datasets/ShufanSun/harmony) ·
[project page](https://cwchenwang.github.io/harmony/)). This is the harness
that produces the leaderboard numbers.

Two families of metrics:

- **Appearance**, over all 300 scenes — N-CLIP (1 − CLIP cosine), PL
  (photometric MSE) and LPIPS, each rendered from the scene's own camera
  (`camera_vggt.json`) and compared to the reference photograph.
- **Geometry**, over the 100 Front3D scenes that ship a ground-truth mesh —
  Chamfer distance and F-score at 0.1 / 0.01 / 0.001, after Sim(3)
  normalisation and ICP alignment (methods do not share the GT world frame).

## Files

| File | Role |
|---|---|
| `evaluation.sh` | one-shot appearance run: metrics → CSV → figures |
| `eval_runner.py` | N-CLIP + PL + LPIPS, multi-method |
| `chamfer_eval.py` | Chamfer + F-score vs. Front3D GT meshes (ICP-aligned) |
| `loaders.py` | per-method render lookup (see registry below) |
| `metrics.py` | `clip_similarity`, `photometric_loss`, `lpips_distance` |
| `json_to_csv.py` | flatten eval JSON into a wide CSV (one row per scene) |
| `make_comparison_figures.py` | per-scene side-by-side panels + `index.html` |
| `render_input_view.py` | render a scene from its reference camera |

## Run

### Appearance (all 300 scenes)

Dataset and baseline locations are placeholders (`/path/to/...`) read from env
vars — set them once, then run the one-shot script:

```bash
export REF_DIR=/path/to/harmony300/ref                   # reference photographs
export REGEN_DIR=/path/to/3D-RE-GEN/results
export GEN3DSR_DIR=/path/to/Gen3DSR/out
export HARMONY_ROOTS=/path/to/harmony/outputs            # comma-separated, earlier roots win
bash evaluation.sh
```

Or call the runner directly, with one `--pred NAME=LOADER:DIR` per method:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python eval_runner.py \
    --ref  flat:$REF_DIR \
    --pred Harmony=harmony:$HARMONY_ROOTS \
    --pred Gen3DSR=gen3dsr:$GEN3DSR_DIR \
    --pred 3DREGEN=3dregen:$REGEN_DIR \
    --out  results_appearance.json
```

### Geometry (the 100 Front3D scenes)

Method roots are explicit, since each baseline stores meshes differently. Any
`--<method>-root` left unset is skipped; the scene list comes from `--gt-root`,
so every method is scored on the same scenes.

```bash
PYTHONNOUSERSITE=1 python chamfer_eval.py \
    --gt-root      /path/to/front3d_gt/sceneobjgt \
    --harmony-root /path/to/harmony/outputs \
    --out          results_geometry.json
```

Needs a CUDA GPU (pytorch3d ICP). It aborts immediately on an unrecoverable
GPU fault rather than writing a file full of identical errors.

## Loader registry (`loaders.py:LOADERS`)

Each loader takes a root path and returns a `{scene_key: image_path}` map;
keys are normalised (`,` → `_`, spaces → `_`).

| name | layout |
|---|---|
| `flat` | `{root}/{stem}.{ext}` — reference photos, accepts jpg/png/avif/webp |
| `harmony` | `{root}/{scene}/decorations/placements/render_*.png`, with fallbacks |
| `harmony_blender`, `harmony_ambient`, `harmony_unlit`, `harmony_input` | other render variants of the same tree |
| `harmony_astra`, `harmony_astra_all` | GPT-Astra bundles, `rgb_` id mapped via `attribution.csv` |
| `3dregen` | `{root}/{scene}/render_cam1_white_bg.png` |
| `gen3dsr` | `{root}/{scene}/render_bproc.png` → `render_input_view.png` |
| `sam3d`, `sam3d_texbg`, `sam3d_rawnobg`, `sam3d_room` | SAM3D variants; `_room` is the room-composited render |
| `cast`, `cast_room`, `cast_reproj` | CAST variants |
| `viga` | VIGA renders |

### Adding a method

1. Write `load_xxx(root: str) -> dict[str, str]` in `loaders.py`.
2. Register it in the `LOADERS` dict at the bottom.
3. Pass `--pred NewName=xxx:/path/to/dir` to `eval_runner.py`.
4. For the figures, add it to `make_comparison_figures.py:methods`.

## Metric semantics

| Metric | Direction | What it measures |
|---|---|---|
| `n_clip` | ↓ | `1 − cosine(CLIP(pred), CLIP(ref))`, CLIP-ViT-B/32 |
| `pl` | ↓ | mean pixel MSE in normalised RGB ∈ [0,1] |
| `lpips` | ↓ | LPIPS perceptual distance |
| `Chamfer` | ↓ | symmetric point-cloud distance after ICP |
| `F1@t` | ↑ | F-score at distance threshold `t` |

If pred and ref differ in size, `pl` is computed on the resized pair (a warning
is printed); aspect ratio is preserved.

## Footguns

- **Always export `PYTHONNOUSERSITE=1`.** Otherwise
  `~/.local/lib/python3.10/site-packages` torch / transformers can shadow the
  intended env and either crash (`libcudart.so.12` missing) or return the wrong
  shapes from `get_image_features`.
- **CLIP cosine is compressed.** Unrelated images still score ≈ 0.9+ on
  CLIP-B/32, so `n_clip` lands in a narrow band for every method. Compare
  deltas between methods, not absolute values.
- **AVIF inputs** need `pillow-avif-plugin` installed in the env.
- **Coverage is not uniform.** A method scored on a subset is not comparable to
  one scored on all 300; `eval_runner.py` reports `num_pairs` per method, and
  the leaderboard marks partial coverage rather than hiding it.
