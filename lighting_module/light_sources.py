"""Collect light-emitting objects from placement JSONs.

Returns a list of `LightSourceRef` records:
  kind        — "lamp" | "ceiling_fixture" | "window"
  index       — original placement index in its source JSON
  source      — "decorations" | "wall_mounted"
  glb_path    — absolute path to the GLB
  inpaint     — absolute path to the per-object inpaint PNG (or None)
  position_m  — world-space anchor (3,)
  rotation_3x3 — placement rotation (3, 3) np.ndarray, identity for windows
  size_m      — {width_m, height_m, depth_m}
  raw         — the original placement dict (for downstream metadata)

The classifier is deliberately conservative: it only marks objects whose
type/phrase string clearly indicates an emissive role.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

# Substrings (lowercase) that mark a placement as a light source
_LAMP_TOKENS = ("lamp", "lantern", "candle")
_CEILING_TOKENS = ("chandelier", "pendant", "ceiling_light", "ceiling light",
                   "sconce", "wall_light", "wall light", "spotlight",
                   "track_light", "track light", "recessed", "downlight",
                   "flush mount", "dome light", "ceiling lamp")
_WINDOW_TOKENS = ("window", "skylight", "french_door", "glass_door",
                  # a curtain/drape covers a window → it IS the daylight source
                  "curtain", "drape", "sheer", "blinds")


def _matches(text: str, tokens: Iterable[str]) -> bool:
    t = text.lower()
    return any(tok in t for tok in tokens)


def _classify(type_str: str, phrase: str) -> str | None:
    blob = f"{type_str} {phrase}".lower()
    # Ceiling tokens first: a chandelier/pendant phrase often also contains
    # "lamp" (e.g. "chandelier pendant lamp"), which must not fall through to
    # the generic lamp bucket — ceiling-mount wording is more specific.
    if _matches(blob, _CEILING_TOKENS):
        return "ceiling_fixture"
    if _matches(blob, _LAMP_TOKENS):
        return "lamp"
    if _matches(blob, _WINDOW_TOKENS):
        return "window"
    # A bare "light" (a wall sconce typed simply "light", a floor/reading
    # "light", …) is still an emitter — treat it as a point-light lamp.  Runs
    # last so "ceiling light"/"window"/"lamp" keep their specific kinds.
    if "light" in blob:
        return "lamp"
    return None


def _abs_glb(out_dir: Path, source_dir: Path, entry: dict) -> Path | None:
    """Resolve a placement's GLB path against the run directory."""
    raw = entry.get("glb_path") or entry.get("glb_file") or ""
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    # decorations entries store both glb_file (relative to out_dir/decorations/)
    # and glb_path (relative to repo root). Try both anchors.
    for anchor in (out_dir, source_dir, source_dir / "objects",
                   out_dir / "decorations" / "objects",
                   out_dir / "wall_mounted" / "objects"):
        cand = anchor / Path(raw).name
        if cand.exists():
            return cand
        cand2 = anchor / raw
        if cand2.exists():
            return cand2
    return None


def _abs_inpaint(source_dir: Path, entry: dict) -> Path | None:
    name = entry.get("inpaint_file")
    if not name:
        return None
    for cand in (source_dir / "inpainted" / name,
                 source_dir / "rectified" / name,
                 source_dir / "segmented" / name):
        if cand.exists():
            return cand
    return None


def collect(output_dir: str | Path) -> list[dict]:
    """Walk decoration + wall-mounted placements and return light-source records.

    The returned dicts are JSON-friendly (paths are stringified) so they can
    be embedded directly in `lights.json`.
    """
    out_dir = Path(output_dir)
    refs: list[dict] = []

    # Decorations: lamps + ceiling fixtures
    dec_json = out_dir / "decorations" / "placements" / "decoration_placements.json"
    if dec_json.exists():
        dec_dir = out_dir / "decorations"
        for e in json.loads(dec_json.read_text()):
            kind = _classify(e.get("type", ""), e.get("phrase", ""))
            if kind not in ("lamp", "ceiling_fixture"):
                continue
            glb = _abs_glb(out_dir, dec_dir, e)
            if glb is None:
                print(f"[lighting] skip {e.get('phrase')!r} — GLB not found")
                continue
            refs.append({
                "kind": kind,
                "source": "decorations",
                "index": int(e.get("index", e.get("seg_index", 0))),
                "phrase": e.get("phrase", e.get("type", "lamp")),
                "glb_path": str(glb),
                "inpaint": str(_abs_inpaint(dec_dir, e)) if _abs_inpaint(dec_dir, e) else None,
                "position_m": list(map(float, e["position_m"])),
                "rotation_3x3": np.array(e["rotation_3x3"], dtype=float).tolist(),
                "size_m": dict(e.get("size_m") or {}),
                "raw": e,
            })

    # Wall-mounted: windows, sconces, ceiling lights placed against walls
    wm_json = out_dir / "wall_mounted" / "placements" / "object_placements.json"
    if wm_json.exists():
        wm_dir = out_dir / "wall_mounted"
        for e in json.loads(wm_json.read_text()):
            kind = _classify(e.get("type", ""), e.get("type", ""))
            if kind is None:
                continue
            glb = _abs_glb(out_dir, wm_dir, e)
            # Windows always count as light sources even without a GLB; an
            # invisible directional light still works.
            if glb is None and kind != "window":
                print(f"[lighting] skip wall {e.get('type')!r} — GLB not found")
                continue
            refs.append({
                "kind": kind,
                "source": "wall_mounted",
                "index": int(e.get("segment_index", 0)),
                "phrase": e.get("type", kind),
                "wall": e.get("wall"),
                "glb_path": str(glb) if glb is not None else None,
                "inpaint": str(_abs_inpaint(wm_dir, e)) if _abs_inpaint(wm_dir, e) else None,
                "position_m": list(map(float, e["world_pt"])),
                "rotation_3x3": np.eye(3).tolist(),
                "size_m": dict(e.get("size_m") or {}),
                "raw": e,
            })

    # Furniture lamps: floor / table / reading lamps placed by the furniture
    # phase.  furniture_placements.json carries index/type/position_m (no
    # glb_file / rotation) — the GLB is furniture/objects/inpaint_<idx>_<type>.glb.
    fur_json = out_dir / "furniture" / "furniture_placements.json"
    if fur_json.exists():
        fur_dir = out_dir / "furniture"
        for e in json.loads(fur_json.read_text()):
            if e.get("is_carpet") or "position_m" not in e:
                continue
            kind = _classify(e.get("type", ""), e.get("type", ""))
            if kind != "lamp":
                continue
            idx = int(e.get("index", 0))
            typ = str(e.get("type", "lamp"))
            glb = fur_dir / "objects" / f"inpaint_{idx:02d}_{typ}.glb"
            if not glb.exists():
                cand = sorted(fur_dir.glob(f"objects/inpaint_{idx:02d}_*.glb"))
                glb = cand[0] if cand else None
            if glb is None or not glb.exists():
                print(f"[lighting] skip furniture {typ!r} — GLB not found")
                continue
            inp = fur_dir / "inpainted" / f"inpaint_{idx:02d}_{typ}.png"
            refs.append({
                "kind": kind,
                "source": "furniture",
                "index": idx,
                "phrase": typ.replace("_", " "),
                "glb_path": str(glb),
                "inpaint": str(inp) if inp.exists() else None,
                "position_m": list(map(float, e["position_m"])),
                "rotation_3x3": np.eye(3).tolist(),
                "size_m": {},
                "raw": e,
            })

    # Ceiling fixtures: track lights, pendants, chandeliers placed by the
    # ceiling phase.  These live in ceiling/segment_results.json (each segment
    # carries a `placement` with the 3D position at the ceiling), not in a
    # decoration/wall-mounted placement JSON — so collect them here or the main
    # interior light source is silently missed and the room renders dim.
    ceil_json = out_dir / "ceiling" / "segment_results.json"
    if ceil_json.exists():
        ceil_dir = out_dir / "ceiling"
        data = json.loads(ceil_json.read_text())
        for seg in (data.get("segments") or []):
            kind = _classify(seg.get("type", ""), seg.get("phrase", ""))
            if kind != "ceiling_fixture":
                continue
            plc = seg.get("placement") or {}
            if "position_m" not in plc:
                continue
            gf = seg.get("glb_file")
            glb = ceil_dir / gf if gf else None
            if glb is None or not glb.exists():
                print(f"[lighting] skip ceiling {seg.get('type')!r} — GLB not found")
                continue
            inp = seg.get("inpaint_file")
            inp_path = ceil_dir / "inpainted" / inp if inp else None
            refs.append({
                "kind": kind,
                "source": "ceiling",
                "index": int(seg.get("index", 0)),
                "phrase": seg.get("phrase", seg.get("type", "ceiling light")),
                "glb_path": str(glb),
                "inpaint": str(inp_path) if inp_path and inp_path.exists() else None,
                "position_m": list(map(float, plc["position_m"])),
                "rotation_3x3": np.array(plc.get("rotation_3x3", np.eye(3)),
                                         dtype=float).tolist(),
                "size_m": dict(seg.get("size_m") or {}),
                "raw": seg,
            })

    return refs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    print(json.dumps(collect(a.output_dir), indent=2, default=str))