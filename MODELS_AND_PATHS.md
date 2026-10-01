# Models & paths you need to fill in

This copy of the repo has the runtime source only. A few things live outside
it — either because they're large vendored model repos, actual input data, or
per-user secrets. Each has a placeholder marking exactly where it goes.

| What | Where | Required? | Notes |
|---|---|---|---|
| VGGT | `vggt/` (directly under the repo root — hardcoded path) | **Required** | `git clone https://github.com/facebookresearch/vggt.git vggt` (see `vggt/PLACE_MODEL_HERE.md`). Weights (`facebook/VGGT-1B`) auto-download from Hugging Face on first run. |
| Grounded-Segment-Anything | `Grounded-Segment-Anything/` (directly under the repo root — hardcoded path) | **Required** | `git clone --recurse-submodules https://github.com/IDEA-Research/Grounded-Segment-Anything.git`, then put its two default checkpoints in `Grounded-Segment-Anything/weights/`: `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` and `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`. Still used for the SAM ViT-H / GroundingDINO weights under `weights/` that `wall_mounted/rectify_windows.py`, `wall_mounted/generate_window_plane.py`, `wall_mounted/segment_wall_objects.py` (whose helpers the ceiling stage imports) and `missing_items/place_missing_items.py` load directly, even though wall/furniture/decoration segmentation itself moved to LocateAnything→SAM2. |
| Hunyuan3D-2 | `Hunyuan3D-2/` | **Required** | `git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git Hunyuan3D-2`, then follow its README for the `hunyuan3d` env (`api_server.py` defaults auto-download from Hugging Face: shape `tencent/Hunyuan3D-2mini`, texture `tencent/Hunyuan3D-2`). See `Hunyuan3D-2/PLACE_MODEL_HERE.md`. The only 3D-generation backend, so its server must be running for the furniture/decoration/wall-mounted/ceiling object-generation stages. Those stages can be skipped with `--skip` while iterating. |
| Qwen3-VL-30B-A3B-Instruct-FP8 (the VLM, port 8080) | *(not a repo path — HF model id)* | **Required** | Served via `vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 ...` (see README `## Backends`) — vLLM pulls it from Hugging Face itself, nothing to copy in-tree. Or skip local vLLM entirely and set `SCENEWEAVE_VLM_BACKEND=gpt` in `.env` to use a hosted VLM API instead. |
| Qwen-Image-Edit (image editing/inpainting, port 8000) | *(not a repo path — HF model id `Qwen/Qwen-Image-Edit`)* | **Required** (or Gemini) | Loaded by `Qwen/img_server.py` through diffusers' `QwenImageEditPipeline`, which pulls it from Hugging Face on the server's first start (<https://huggingface.co/Qwen/Qwen-Image-Edit>). Not the same model as the Qwen3-VL VLM above. Or set `SCENEWEAVE_IMG_EDIT=gemini` with `GEMINI_API_KEY`. |
| Input photos | `data/indoor_images/` | **Required** | See `data/PLACE_DATA_HERE.md`. |
| API keys (Gemini / NVIDIA-hosted VLM / OpenAI eval) | `.env` (copy from `.env.example`) | Only if using hosted backends | `cp .env.example .env` then fill in. `.env` is gitignored. |
| VLM + image-edit servers env | conda env `vllm-env` | **Required** (for local backends) | Spec at `environment_yml/vllm-env.yml`. |
| Main pipeline env | conda env `scenegen` | **Required** | `requirements.txt` lists the pip packages; there is no exported conda spec yet (needs `conda env export` from a working install). |
| LocateAnything segmentation env | conda env `eagle-embodied` | **Required** | No exported spec yet. See `object_placement/segmentation_la_sam2.py` header. |
| SAM2 segmentation env | conda env `lang-sam` | **Required** | No exported spec yet. Conflicts with `eagle-embodied`'s transformers version, hence the separate env. |
| Hunyuan3D server env | conda env `hunyuan3d` | **Required** | No exported spec yet — comes from Hunyuan3D-2's own setup instructions. |
| SAM ViT-H checkpoint (carpet segmentation fallback) | `checkpoints/sam_vit_h_4b8939.pth` at repo root, or `SCENEWEAVE_SAM_CKPT` env var | Optional | Only reached when the pre-existing carpet mask is missing in `place_furniture_vggt.py`; silently skipped if absent. |
| LocateAnything + SAM2 workers | `object_placement/la_sam2_workers/` (in-tree; `SCENEWEAVE_LA_DIR` / `SCENEWEAVE_SAM2_DIR` override) | **Required** | See `object_placement/la_sam2_workers/README.md` for where each model comes from. |
| LocateAnything-3B | *(not a repo path — HF model id `nvidia/LocateAnything-3B`)* | **Required** | The object detector for wall, furniture and decoration segmentation. Auto-downloads from Hugging Face on first run, or `hf download nvidia/LocateAnything-3B` and set `SCENEWEAVE_LA_MODEL=<dir>`. Can be replaced via `la_emit_boxes.py` (see its header). |
| NVlabs/Eagle `Embodied/` (`locateanything_worker.py`) | `SCENEWEAVE_EAGLE_EMBODIED_DIR` env var | **Required** | Clone <https://github.com/NVlabs/Eagle>; its `Embodied/` setup also defines the `eagle-embodied` env. |
| SAM 2.1 (`sam2.1_hiera_small`) | *(not a repo path — torch hub cache)* | **Required** | Masks only: turns LocateAnything's boxes into pixel masks and does no detection. Can be replaced by any box-prompted segmenter. Auto-downloaded by `lang_sam` from Meta's `dl.fbaipublicfiles.com`. Install `lang_sam` from <https://github.com/luca-medeiros/lang-segment-anything> into the `lang-sam` env. |
| Evaluation inputs (baselines + input photos) | `REF_DIR`, `REGEN_DIR`, `GEN3DSR_DIR`, `HARMONY_ROOTS` env vars (+ `OPENAI_API_KEY`) | Optional | Only for `evaluation/evaluation.sh`; see `evaluation/README.md`. |
| Blender (Cycles lighting render) | `BLENDER_BIN` env var; otherwise `_find_blender()` in `lighting_module/render_blender.py` globs `~/blender/blender-*/blender`, `/opt/blender*/blender`, `/usr/local/blender*/blender`, then falls back to `blender` on `PATH` | Optional | Only needed for the physically-lit `lighting` stage's Cycles render; the pipeline degrades gracefully without it. |

## Not carried over from the working repo

- `scratch/` — one-off experiment scripts, not needed to run the pipeline.
- `supplementary/` — paper LaTeX/figures/website, unrelated to the runtime code.
- the working repo's old `main.py` (v1 pipeline) and `main_frist2stages.py` —
  both superseded by the v2 pipeline, which is the `main.py` in this copy
  (renamed from `main_v2.py` for a cleaner release entry point).
- Depth estimation is VGGT-only (`vggt/`); no separate monocular depth model
  is shipped. `Grounded-Segment-Anything/` is NOT dead — see the table above.
- `depth_alignment/` (top-level) — small, unreferenced by any current
  pipeline stage. `dataset_generation/image_fetcher.py` was NOT dropped —
  it's a genuine data-downloading tool, so it moved to `data/image_fetcher.py`
  instead (see `data/PLACE_DATA_HERE.md`).
- Dead/orphaned per-category segmentation modules superseded by
  `segmentation_la_sam2.py` but never deleted from the working repo:
  `object_placement/segmentation_sceneconductor.py` (an alternate VLM+Grounded-SAM
  front-end, never wired into `main.py`), `object_placement/furniture/segment_furniture.py`,
  `object_placement/decorations/segment_decorations.py` and `segment_decorations_vlm.py`,
  `object_placement/wall_mounted/segment_wall_objects_wip.py` (orphaned draft),
  `floorplan/vggt_estimates/manhattan_pre_mod.py`.
  Note: `object_placement/wall_mounted/segment_wall_objects.py` itself is KEPT —
  it looked like the same class of leftover but is a live dependency (the
  ceiling stage imports its GroundingDINO helpers directly).
- The secondary 3D-generation worker directory and its `agent` conda env — not
  carried over. Hunyuan3D is the only 3D-generation backend for
  furniture/decoration/wall-mounted/ceiling object generation; on a Hunyuan3D
  404 (the known can't-mesh-this signature, typically glass/transparent
  objects), the pipeline re-inpaints that one object as an opaque material and
  retries once, instead of falling back to a second backend.
- `phys_sim/` — Isaac Sim / earthquake-physics demo tooling, standalone and
  never part of `main.py`'s pipeline stages.

## Removed during release cleanup

Dead code that had been carried over and has since been deleted from this copy
(recoverable from the `Initial import` commit):

- `Qwen/img_edit.py` — a hardcoded one-off dev scratch script.
- `floorplan/wall_line/deepest_corner.py` — needed Marigold depth maps, which
  are not shipped (depth is VGGT-only).
- `object_placement/furniture/detilt_and_orient.py` — unimportable: its
  top-level import referenced `place_furniture.py`, which was never carried
  over. Nothing imported or subprocessed it.
- `object_placement/furniture/place_carpet.py` — an orphaned alternative
  carpet path. The live path is the flat-quad / `world_corners` one inside
  `place_furniture_vggt.py` plus `lighting_module/bake_carpet.py`.
- 60 top-level functions whose names appeared exactly once in the whole tree
  (their own `def`), spread over 20 modules.
- The top-down render block in `place_furniture_vggt.py`, which imported
  `scripts/top_down_render.py` — not shipped — inside a bare `try/except`.

## Known gaps

- Only `vllm-env` has an exported conda spec. `requirements.txt` covers the pip
  side of `scenegen`; `eagle-embodied`, `lang-sam` and `hunyuan3d` have none.
- There are no tests.

## First run checklist

```bash
# 1. Vendored repos — clone all three per their PLACE_MODEL_HERE.md files.
#    vggt/ and Grounded-Segment-Anything/ are imported off sys.path;
#    Hunyuan3D-2/ is the 3D-generation server.

# 2. Main pipeline env. Install the CUDA torch/torchvision wheels matching
#    your driver FIRST (https://pytorch.org), then the rest.
conda create -n scenegen python=3.10
conda activate scenegen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt

# 3. The other four envs: vllm-env (spec at environment_yml/vllm-env.yml),
#    hunyuan3d, eagle-embodied, lang-sam. See the table above.
conda env create -f environment_yml/vllm-env.yml

# 4. Config. Nothing sources .env automatically — export it yourself in every
#    shell that runs the pipeline.
cp .env.example .env          # fill in keys and the two segmentation dirs
set -a; source .env; set +a

# 5. Start the three servers, each in its own terminal (see README "## Backends").

# 6. Run.
python main.py --image data/indoor_images/<your_photo>.jpg
```
