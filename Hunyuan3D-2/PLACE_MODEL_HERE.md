# Hunyuan3D-2 — required, clone here

The **only** 3D-generation backend in this release.
(`object_generation`, `furniture_object_generation`,
`decoration_object_generation`, `ceiling_object_generation`) — without it those
stages fail and nothing downstream of them gets placed.

Not imported into the Python path — the pipeline talks to it over HTTP at
`HUNYUAN_SERVER` (default `http://localhost:8081`).

Clone it into this exact folder (replacing this placeholder):

```bash
cd <repo_root>
rm -rf Hunyuan3D-2   # remove this placeholder dir first
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git Hunyuan3D-2
# follow Hunyuan3D-2's own setup instructions for the `hunyuan3d` conda env + weights
```

Then start its server in its own terminal / conda env before running the
pipeline:

```bash
conda activate hunyuan3d
cd Hunyuan3D-2
python api_server.py --host 0.0.0.0 --port 8081 --enable_tex
```

On a Hunyuan3D 404 — the known can't-mesh-this signature, typically a glass or
transparent object deadlocking marching cubes — the pipeline re-inpaints that
one object as an opaque material and retries once before giving up on it.

Used by: `object_placement/furniture/object_generation.py` (the shared
Hunyuan3D client, also used by the wall-mounted and ceiling stages),
`object_placement/decorations/generate_decoration_3d.py`.
