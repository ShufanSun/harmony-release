"""
anchor_seating_group.py — standalone runner for the seating-group anchor rule
(central coffee table → in front of sofa, long side parallel; accent chairs →
face the table, off the camera plane).

The logic lives in object_placement/furniture/seating_group.py and runs
automatically inside place_furniture_vggt.run(); this script just applies it to
an already-placed scene's furniture_placements.json (useful for re-deriving the
group without a full re-placement).

Usage:
    python scripts/anchor_seating_group.py --scene outputs/front3d/rgb_003084
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from object_placement.furniture import seating_group as sg


def run(scene: str | Path) -> None:
    scene = Path(scene)
    pj = scene / "furniture" / "furniture_placements.json"
    placements = json.load(open(pj))
    cam = json.load(open(scene / "camera.json"))
    sg.anchor_group(placements, cam, scene)
    json.dump(placements, open(pj, "w"), indent=1)
    print(f"[anchor] wrote {pj}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    run(ap.parse_args().scene)
