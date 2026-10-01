# LocateAnything → SAM2 segmentation workers

`object_placement/segmentation_la_sam2.py` runs these as subprocesses, each in
its own conda env, for the `segment_wall_objects`, `segment_furniture` and
`segment_decorations` stages. The two envs pin conflicting `transformers`
versions, so they can't be merged.

**NVIDIA LocateAnything-3B is the detector.** It decides what objects exist
and where they are. SAM2 does no detection: it only turns each LocateAnything
box into a pixel mask, because LocateAnything outputs boxes and points, not
masks, and the inpainting and 3D-generation stages need masks.

Both models can be replaced. The two stages communicate only through
`boxes.json`, so you can swap in another detector (keep the `boxes.json`
schema) or another box-prompted segmenter (return one mask per box). Each
script's header says exactly what to replace.

## Downloading the models

```bash
# 1. LocateAnything: wrapper code + eagle-embodied env
git clone https://github.com/NVlabs/Eagle && cd Eagle/Embodied
conda create -n eagle-embodied python=3.10 -y && conda activate eagle-embodied
pip install -e .                      # pins transformers==4.57.1
export SCENEWEAVE_EAGLE_EMBODIED_DIR=$PWD
# weights: auto-download on first run, or fetch ahead of time (offline nodes):
hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
export SCENEWEAVE_LA_MODEL=$PWD/LocateAnything-3B   # optional; default is the HF id

# 2. SAM2 (masks only): lang_sam + lang-sam env
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
conda create -n lang-sam python=3.11 -y && conda activate lang-sam
pip install -e .
# sam2.1_hiera_small.pt auto-downloads to the torch hub cache on first run
```

| File | Runs in | What it does |
|---|---|---|
| `la_emit_boxes.py` | `eagle-embodied` | Stage 1: LocateAnything detection, then VLM type discovery and verification, then writes `boxes.json` |
| `la_pipeline_segment.py` | `eagle-embodied` | Helpers that `la_emit_boxes.py` imports (type lists, box parsing, VLM verify). Its own `main()` is an older single-env LA→SAM-ViT-H path that the pipeline doesn't call |
| `sam2_from_boxes.py` | `lang-sam` | Stage 2: box-prompted SAM2 masks, then writes HARMONY's `segment_results.json` |

## Models and where they come from

| Model | Used by | Source | How it gets onto disk |
|---|---|---|---|
| **LocateAnything-3B** | `la_emit_boxes.py` (`--model`, default `nvidia/LocateAnything-3B`) | NVIDIA, Hugging Face: <https://huggingface.co/nvidia/LocateAnything-3B> | Downloaded into the HF cache by `transformers` (`trust_remote_code=True`) on first run. Or pass `--model /local/path` |
| **LocateAnything inference wrapper** (`locateanything_worker.py`) | `la_pipeline_segment.py` | `Embodied/` folder of NVIDIA Eagle: <https://github.com/NVlabs/Eagle> (Apache-2.0). We use it unmodified, so it isn't copied here | Clone NVlabs/Eagle, build the `eagle-embodied` env from its `Embodied/` instructions, and `export SCENEWEAVE_EAGLE_EMBODIED_DIR=/path/to/Eagle/Embodied` |
| **SAM 2.1** (`sam2.1_hiera_small` by default; set `SCENEWEAVE_SAM2_TYPE` for tiny/base_plus/large) | `sam2_from_boxes.py` | Meta, <https://github.com/facebookresearch/sam2>. Checkpoints come from `https://dl.fbaipublicfiles.com/segment_anything_2/092824/` | Downloaded into the torch hub cache by `lang_sam` on first run |
| **lang-segment-anything** (`lang_sam` package, SAM2 wrapper) | `sam2_from_boxes.py` | <https://github.com/luca-medeiros/lang-segment-anything> (we used v0.2.1) | `pip install -e .` from a clone, inside the `lang-sam` env |
| **VLM for type discovery and box verification** | `la_pipeline_segment._discover_types` / `_verify` | The pipeline's shared VLM backend, `object_placement/vlm_backend.py`: by default Qwen3-VL-30B-A3B-Instruct-FP8 served by vLLM on :8080, or a hosted GPT model with `SCENEWEAVE_VLM_BACKEND=gpt` | Same server as the rest of the pipeline (see the main README) |

### Silent fallbacks

- If the VLM is unreachable, both VLM steps fail silently. Discovery returns no
  extra types, and verification keeps every box (logged as `SKIP(...)`).
  Segmentation still runs, but with lower recall and no false-positive filtering.
  If your log shows `SKIP(no-backend:...)` or `SKIP(error:...)` verdicts, the
  VLM is not being reached.
- SAM ViT-H (`Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth`) is
  only loaded by `la_pipeline_segment.py`'s standalone `main()`, not by the
  pipeline.
