"""Lighting stage — VLM-driven environment + per-source light estimation.

Reads existing placements from `wall_mounted/` and `decorations/`, asks the
VLM to estimate ambient + directional + point lights from the reference
photo, edits emissive properties on light-source GLBs (lamps, windows) into
copies stored under `<output>/lightings/`, renders a lit preview, and runs
an iterative VLM refinement loop. Other stage outputs are never modified.
"""

from .pipeline import run

__all__ = ["run"]
