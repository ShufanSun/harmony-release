# Input data — put your own images here

- `indoor_images/` — put your room photos here. Used both for single runs
  (`python main.py --image data/indoor_images/<photo>.jpg`) and as the default
  input dir for batch runs (`scripts/batch_main.py`).

Any indoor room photo works (jpg/jpeg/png). Note: `.avif` source images crash
the VLM stages — convert to PNG first.

## Don't have sample photos yet?

`image_fetcher.py` (in this folder) fetches credit-free indoor-corner photos
from a free stock provider and VLM-filters them down to ones that show a
clear room interior with a visible wall corner — exactly what `main.py`'s
floorplan stage needs:

```bash
export PEXELS_API_KEY="your_key"   # or PIXABAY_API_KEY / UNSPLASH_API_KEY
python data/image_fetcher.py
```

Filtered results land in `data/fetched_data/VLM-filtered/`; rejected ones stay
in `data/fetched_data/raw/` for inspection. Copy (or `--output-root data`
symlink) the accepted photos into `indoor_images/` to run them. See the script's own docstring for
the full option list (`--provider`, `--query`, `--count`, `--output-root`).

Everything else under `data/` is gitignored (actual photos aren't meant to be
committed) — `image_fetcher.py` and this file are the two exceptions kept
tracked in git.
