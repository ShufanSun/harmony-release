"""Render a GLB from the original input-photo camera viewpoint.

Used after patching SceneGen (`--keep_input_camera`) or MIDI's
`image_to_textured_scene.py` so that their output GLBs preserve the
input camera frame.

Conventions assumed:
    - GLB vertices are in the input camera world frame (Y-up).
    - For MIDI: camera at origin, looking down -Z (OpenGL).
    - For SceneGen: camera at origin with Euler (90, 0, 0) — equivalent to
      "looking down +Y after a 90 deg X rotation". A sidecar JSON
      <glb>_cam.json with `camera_rotation_euler_xyz_deg` is read when present.

Usage:
    python render_input_view.py <glb> [<png_out>]
    python render_input_view.py --batch <dir>   # renders every glb in dir
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R


def load_camera_cfg(glb_path):
    """Read <glb>_cam.json if it exists; else return MIDI defaults."""
    cfg_path = glb_path.replace(".glb", "_cam.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        # MIDI default: scene is roughly in [-1, 1]^3 around origin. MIDI does
        # not preserve the input camera pose; the predicted scene sits in a
        # canonical frame. Use a photographer-like default: camera pulled back
        # along +Z, slightly elevated (+Y), and tilted ~10 deg down toward the
        # scene centroid. This matches the typical "slight downward" angle of
        # indoor photos. Override via <glb>_cam.json if a per-scene pose is
        # available (e.g. from VGGT / Grounded camera estimation).
        cfg = {
            "camera_pos": [0.0, 0.5, 2.5],
            "camera_rotation_euler_xyz_deg": [-10.0, 0.0, 0.0],
            "fov": 0.7,  # ~40 deg
            "width": 1024,
            "height": 1024,
        }
    return cfg


def render(glb_path, out_path, cfg=None):
    cfg = cfg or load_camera_cfg(glb_path)

    # ---- Load mesh ----
    loaded = trimesh.load(glb_path, force="scene", process=False)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

    # ---- Build camera pose (4x4 in OpenGL convention) ----
    # SceneGen's quat_query = Euler(90, 0, 0) describes the camera's orientation
    # in the GLB's world frame. pyrender expects the camera-to-world matrix
    # (OpenGL: camera looks down its -Z axis, +Y up). The same Euler convention
    # is therefore reused directly.
    rot = R.from_euler("xyz",
                       cfg.get("camera_rotation_euler_xyz_deg", [0, 0, 0]),
                       degrees=True).as_matrix()
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = rot
    cam_pose[:3, 3] = np.array(cfg.get("camera_pos", [0, 0, 0]))

    w = int(cfg.get("width", 1024))
    h = int(cfg.get("height", 1024))
    fov_rad = float(cfg.get("fov", 0.7))   # horizontal FoV in radians

    # ---- Scene ----
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0],
                           ambient_light=[0.4, 0.4, 0.4])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    # Convert horizontal FoV to yfov (pyrender uses vertical FoV)
    yfov = 2 * np.arctan(np.tan(fov_rad / 2) * (h / w))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=w / h,
                                     znear=0.01, zfar=100.0)
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0),
              pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, _ = r.render(scene)
    r.delete()
    Image.fromarray(color).save(out_path)
    return f"{w}x{h}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", help="single glb path")
    p.add_argument("out", nargs="?", help="png output path")
    p.add_argument("--batch", help="render every glb in this directory")
    p.add_argument("--out_dir", help="batch output directory")
    args = p.parse_args()

    if args.batch:
        out_dir = args.out_dir or os.path.join(args.batch, "input_view_renders")
        os.makedirs(out_dir, exist_ok=True)
        glbs = sorted(f for f in os.listdir(args.batch) if f.endswith(".glb"))
        for f in glbs:
            inp = os.path.join(args.batch, f)
            out = os.path.join(out_dir, f.replace(".glb", ".png"))
            try:
                msg = render(inp, out)
                print(f"[OK] {f}: {msg}")
            except Exception as e:
                print(f"[FAIL] {f}: {type(e).__name__}: {e}")
    else:
        if not args.path:
            p.error("provide a glb path or use --batch <dir>")
        out = args.out or args.path.replace(".glb", "_inputview.png")
        msg = render(args.path, out)
        print(f"[OK] {args.path} -> {out} ({msg})")


if __name__ == "__main__":
    main()
