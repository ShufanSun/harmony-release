"""
resolve_furniture_collisions.py — standalone runner for the furniture collision
resolver + height cap.

The logic lives in object_placement/furniture/seating_group.py and runs
automatically inside place_furniture_vggt.run(); this script just applies it to
an already-placed scene's furniture_placements.json.

Usage:
    python scripts/resolve_furniture_collisions.py --scene outputs/front3d/rgb_003084
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from object_placement.furniture import seating_group as sg


def run(scene: str | Path) -> None:
    scene = Path(scene)
    pj = scene / "furniture" / "furniture_placements.json"
    placements = json.load(open(pj))
    v = np.array([[float(x) for x in l.split()[1:4]]
                  for l in open(scene / "walls.obj") if l.startswith("v ")])
    room_w, room_d = float(v[:, 0].max()), float(v[:, 2].max())
    sg.resolve_collisions(placements, room_w, room_d)
    json.dump(placements, open(pj, "w"), indent=1)
    print(f"[collision] wrote {pj}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    run(ap.parse_args().scene)
