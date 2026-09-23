# VGGT — required, clone here

`floorplan/vggt_estimates/inference.py` expects the VGGT repo **directly under
the repo root**, i.e. exactly `<repo_root>/vggt/`:

```python
_VGGT_REPO = Path(__file__).parent.parent.parent / "vggt"
```

Clone it into this exact folder (replacing this placeholder):

```bash
cd <repo_root>
rm -rf vggt   # remove this placeholder dir first
git clone https://github.com/facebookresearch/vggt.git vggt
pip install -r vggt/requirements.txt
```

Model weights are pulled automatically from Hugging Face on first run
(`facebook/VGGT-1B`) — no manual download needed.

Used by: `floorplan/vggt_estimates/inference.py`, `floorplan/vggt_estimates/manhattan.py`,
`object_placement/furniture/vggt_refine.py`.
