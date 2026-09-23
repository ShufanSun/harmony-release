"""
Rectangular room mesh generator.

Builds a 4-wall + floor box from the VLM floor dimensions (width, depth,
ceiling height) and exports it as OBJ (+ optional MTL with texture paths).
The box is only the Stage-1 seed: VGGT (Stage 2 Manhattan alignment and
geometry_refine) fits the real wall extents from metric depth.

Usage:
    python wall_mesh_gen.py --width 5.0 --depth 4.0 \
        --out outputs/xxx/walls.obj [--ceiling 2.4]
"""

import argparse
from pathlib import Path

import numpy as np


# ── OBJ writer ────────────────────────────────────────────────────────────────

TILE_SIZE_M = 1.5   # must match render_room.py TILE_SIZE_M


def _face_normal(verts: list, tri: tuple) -> np.ndarray:
    """Compute the face normal for a triangle (OBJ 1-indexed indices)."""
    a = np.array(verts[tri[0] - 1])
    b = np.array(verts[tri[1] - 1])
    c = np.array(verts[tri[2] - 1])
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else n


def _orient_from_normal(n: np.ndarray) -> str:
    ax = int(np.argmax(np.abs(n)))
    if ax == 1:
        return "floor" if n[1] > 0 else "ceiling"
    elif ax == 0:
        return "left" if n[0] > 0 else "right"
    else:
        return "back" if n[2] > 0 else "front"


def _uv_for_vert(xyz: tuple, orient: str) -> tuple[float, float]:
    x, y, z = xyz
    if orient in ("floor", "ceiling"):
        return x / TILE_SIZE_M, z / TILE_SIZE_M
    elif orient in ("left", "right"):
        return z / TILE_SIZE_M, y / TILE_SIZE_M
    else:  # back, front
        return x / TILE_SIZE_M, y / TILE_SIZE_M


def write_obj(path: str, vertices: list, faces: list,
              walls_meta: dict | None = None):
    """
    Write an OBJ file.  When walls_meta is provided (from walls_metadata.json)
    also write a companion .mtl file and embed UV coordinates + material refs
    so the mesh can be opened in any 3-D viewer with textures applied.
    """
    path = str(path)
    obj_path = Path(path)

    write_mtl = walls_meta is not None
    mtl_name  = obj_path.stem + ".mtl"
    mtl_path  = obj_path.parent / mtl_name

    # ── MTL file ──────────────────────────────────────────────────────────────
    orient_to_matname: dict[str, str] = {}
    if write_mtl:
        with open(mtl_path, "w") as mf:
            mf.write("# Material file — wall_mesh_gen.py\n\n")
            seen: dict[str, str] = {}   # tex_path → matname
            for orient in ("back", "front", "left", "right", "floor", "ceiling"):
                info   = walls_meta.get(orient, {})
                tp     = info.get("texture_path") or ""
                if tp and Path(tp).exists():
                    if tp not in seen:
                        mname = f"mat_{orient}"
                        seen[tp] = mname
                        # Use relative path if same directory, else absolute
                        def _tex_ref(p: str) -> str:
                            try:
                                return str(Path(p).relative_to(obj_path.parent))
                            except ValueError:
                                return str(Path(p).resolve())

                        mf.write(f"newmtl {mname}\n")
                        mf.write(f"map_Kd {_tex_ref(tp)}\n")
                        # PBR maps: <stem>_normal/roughness/metallic.png (optional)
                        stem = str(Path(tp).with_suffix(""))
                        for suffix, mtl_key in (
                            ("_normal",    "norm"),
                            ("_roughness", "map_Pr"),
                            ("_metallic",  "map_Pm"),
                        ):
                            pbr_path = stem + suffix + ".png"
                            if Path(pbr_path).exists():
                                mf.write(f"{mtl_key} {_tex_ref(pbr_path)}\n")
                        mf.write("\n")
                    orient_to_matname[orient] = seen[tp]
                else:
                    # Solid colour fallback
                    mname = f"mat_{orient}"
                    col = {
                        "floor":   "0.55 0.47 0.37",
                        "ceiling": "0.92 0.91 0.89",
                    }.get(orient, "0.86 0.83 0.79")
                    mf.write(f"newmtl {mname}\n")
                    mf.write(f"Kd {col}\n\n")
                    orient_to_matname[orient] = mname
        print(f"[wall_mesh] MTL saved → {mtl_path}")

    # ── Pre-compute per-face orientation and UVs ──────────────────────────────
    # Each face vertex gets its own vt entry (no sharing across faces).
    face_orients: list[str] = []
    face_uvs:     list[list[tuple]] = []   # [[uv0,uv1,uv2], ...]
    for tri in faces:
        n      = _face_normal(vertices, tri)
        orient = _orient_from_normal(n)
        face_orients.append(orient)
        uvs = [_uv_for_vert(vertices[i - 1], orient) for i in tri]
        face_uvs.append(uvs)

    # ── OBJ file ──────────────────────────────────────────────────────────────
    with open(obj_path, "w") as f:
        f.write("# Wall mesh — wall_mesh_gen.py\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n")
        if write_mtl:
            f.write(f"mtllib {mtl_name}\n")
        f.write("\n")

        for x, y, z in vertices:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        f.write("\n")

        # vt entries (one per face-vertex, in face order)
        if write_mtl:
            for uvs in face_uvs:
                for u, v in uvs:
                    f.write(f"vt {u:.6f} {v:.6f}\n")
            f.write("\n")

        # Faces grouped by material
        vt_idx   = 1  # running vt index (1-based)
        cur_mat  = None
        for tri, orient, uvs in zip(faces, face_orients, face_uvs):
            if write_mtl:
                mat = orient_to_matname.get(orient, f"mat_{orient}")
                if mat != cur_mat:
                    f.write(f"usemtl {mat}\n")
                    cur_mat = mat
                vt0, vt1, vt2 = vt_idx, vt_idx + 1, vt_idx + 2
                vt_idx += 3
                f.write(f"f {tri[0]}/{vt0} {tri[1]}/{vt1} {tri[2]}/{vt2}\n")
            else:
                f.write("f " + " ".join(str(i) for i in tri) + "\n")

    print(f"[wall_mesh] OBJ saved → {obj_path}  "
          f"({len(vertices)} verts, {len(faces)} faces)"
          + (f"  + MTL" if write_mtl else ""))


# ── full rectangular room from VLM dimensions ─────────────────────────────────

def build_room_mesh(width_m: float, depth_m: float, ceiling_h: float):
    """
    Build a complete rectangular room: 4 walls + floor + ceiling, all sharing edges.

    Coordinate system:
      X  0 → width_m   (left → right)
      Y  0 → ceiling_h (floor → ceiling)
      Z  0 → depth_m   (back wall → front / camera)

    The 8 room corners are shared across all surfaces so there are no gaps.
    """
    W, D, H = width_m, depth_m, ceiling_h

    # 8 shared corner vertices (OBJ 1-indexed)
    #  0: (0,0,0)  1: (W,0,0)  2: (W,0,D)  3: (0,0,D)   — floor ring
    #  4: (0,H,0)  5: (W,H,0)  6: (W,H,D)  7: (0,H,D)   — ceiling ring
    verts = [
        (0, 0, 0), (W, 0, 0), (W, 0, D), (0, 0, D),   # 1-4 floor
        (0, H, 0), (W, H, 0), (W, H, D), (0, H, D),   # 5-8 ceiling
    ]

    # Each quad = two CCW triangles sharing the diagonal
    def quad(a, b, c, d):
        return [(a, b, c), (a, c, d)]

    faces = []
    faces += quad(1, 2, 6, 5)   # back wall   z=0
    faces += quad(2, 3, 7, 6)   # right wall  x=W
    faces += quad(3, 4, 8, 7)   # front wall  z=D
    faces += quad(4, 1, 5, 8)   # left wall   x=0
    faces += quad(1, 4, 3, 2)   # floor       y=0
    faces += quad(5, 6, 7, 8)   # ceiling     y=H

    print(f"[wall_mesh] Room mesh: {W:.2f}m × {D:.2f}m × {H:.2f}m  "
          f"({len(verts)} verts, {len(faces)} faces)")
    return verts, faces


# ── main entry point ──────────────────────────────────────────────────────────

def generate_wall_mesh(out_path: str,
                       ceiling_h: float = 2.4,
                       floor_dims: tuple | None = None,
                       walls_meta: dict | None = None) -> str:
    """
    Generate a complete 4-wall + floor rectangular room from the VLM
    floor_dims=(width_m, depth_m) and ceiling_h, and export it as OBJ.

    walls_meta: if provided, a companion .mtl file is written with texture paths
                so the exported OBJ can be opened with textures in any 3-D viewer.
    """
    if not floor_dims:
        raise ValueError("generate_wall_mesh requires floor_dims=(width_m, depth_m)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fw, fd = floor_dims
    vertices, faces = build_room_mesh(float(fw), float(fd), float(ceiling_h))
    write_obj(out_path, vertices, faces, walls_meta=walls_meta)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width",   type=float, required=True, help="floor width (m)")
    ap.add_argument("--depth",   type=float, required=True, help="floor depth (m)")
    ap.add_argument("--out",     default="outputs/walls.obj")
    ap.add_argument("--ceiling", type=float, default=2.4)
    args = ap.parse_args()

    generate_wall_mesh(out_path=args.out, ceiling_h=args.ceiling,
                       floor_dims=(args.width, args.depth))


if __name__ == "__main__":
    main()
