# Grounded-Segment-Anything — required, clone here

Still a real dependency, despite wall/furniture/decoration segmentation having
moved to the LocateAnything→SAM2 backend (`segmentation_la_sam2.py`): four
modules load the GroundingDINO and SAM ViT-H weights under `weights/` directly.

It is expected **directly under the repo root**, i.e. exactly
`<repo_root>/Grounded-Segment-Anything/`:

```python
_GSA_ROOT   = _REPO_ROOT / "Grounded-Segment-Anything"
_GDINO_CFG  = _GSA_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_GDINO_CKPT = _GSA_ROOT / "weights/groundingdino_swint_ogc.pth"
```

Clone it into this exact folder (replacing this placeholder):

```bash
cd <repo_root>
rm -rf Grounded-Segment-Anything   # remove this placeholder dir first
git clone --recurse-submodules https://github.com/IDEA-Research/Grounded-Segment-Anything.git
# then follow its own setup, which must leave both checkpoints under weights/:
#   weights/groundingdino_swint_ogc.pth   (GroundingDINO SwinT-OGC)
#   weights/sam_vit_h_4b8939.pth          (SAM ViT-H)
```

`groundingdino` and `segment_anything` are imported off `sys.path` from this
clone, not pip-installed into `scenegen`.

Used by:
- `object_placement/wall_mounted/segment_wall_objects.py` — GroundingDINO
  helpers, which `object_placement/ceiling/segment_ceiling_objects.py` imports
- `object_placement/wall_mounted/rectify_windows.py` — SAM
- `object_placement/wall_mounted/generate_window_plane.py` — SAM
- `object_placement/missing_items/place_missing_items.py` — GroundingDINO + SAM
