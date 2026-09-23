"""VLM-driven initial lighting estimate.

Given the reference photo and the list of detected light sources, ask the
VLM for:
  * ambient_light: {color [r,g,b in 0..1], intensity [0..2]}
  * directional_lights: [{name, direction [dx,dy,dz, world space, points
    away from light source towards lit surfaces], color, intensity}]
  * point_lights: per detected lamp/ceiling fixture, the bulb-region offset
    (relative to the lamp's bottom anchor) and the bulb color/intensity.
  * windows: per detected window, daylight color/intensity and whether the
    sun is currently shining through (bool).

The function returns a `lights.json`-shaped dict (validated, with sane
defaults filled in for anything the VLM omits).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .vlm_client import call as _vlm_call

_PROMPT = """\
You are a lighting artist analysing a real-world room photograph so that we \
can re-create its light setup in a 3D scene. You have access to:

IMAGE 1 — the original reference photograph of the room.

The 3D scene already contains every visible light source as a placed object. \
For each one we tell you its world position (metres) and rough physical size, \
so you can reason about which ones are actually emitting light right now.

LIGHT SOURCES IN THE 3D SCENE:
{light_sources_blob}

ROOM CONTEXT:
{room_blob}

TASK — output the lighting setup that, when rendered, would best match the \
reference photo. Use these conventions:

* Colours are normalised RGB triplets in [0, 1].
* Intensity is a small non-negative float. The renderer is sensitive — \
  START SMALL and only raise if a region is clearly under-lit:
    - ambient indoor: 0.05–0.15
    - lamp / point light when ON: 0.10–0.30
    - daylight through a window: 0.15–0.40
  Values above 0.5 are almost always wrong; never exceed 1.0. \
  Set `intensity = 0` and `on = false` for any light source that is not \
  visibly emitting in the reference.
* Directional light direction is the unit vector pointing FROM the light \
  TOWARDS the room (e.g. sun coming through a left-side window into the \
  room would have a direction with negative X component). World axes: \
  +X = room width, +Y = up, +Z = room depth.
* Point lights live at a 3D world point you compute from the lamp's \
  position_m + a small offset that places the bulb inside the lampshade.
* Mark a lamp `on=true` only if its shade/bulb is visibly emitting light in \
  the reference photo (bright, warm, glowing). A lamp that is present but \
  unlit is `on=false` (still listed, intensity 0).

The renderer runs in LIGHTS-ONLY mode by default: the base composite \
already has photographic lighting baked in, so we add NO global wash, \
ambient lift, or tone grade. Each active light only contributes a \
LOCALISED additive glow at its projected screen position:
  * lamp / ceiling fixture → Gaussian halo at the bulb position
  * window → Gaussian halo centred on the window, sized to its physical \
    width and height

Your only job per source is:
  - `on` / `daylight_on` — is it actively emitting in the reference?
  - `color` — warmth/coolness of its emission
  - `intensity` — how strong the localised glow is (small numbers, see \
    above)

Ambient and directional lights are accepted but only contribute in the \
legacy `lights_only=false` path; you may leave them at intensity 0. \
`base_dim` and `grading_strength` are also legacy-path only — leave them \
at their defaults.

OUTPUT — strictly the following JSON object, no markdown, no commentary:

{{
  "ambient": {{"color": [r, g, b], "intensity": 0.0}},
  "directional": [
    {{"name": "sun_through_back_window", "direction": [dx, dy, dz],
      "color": [r, g, b], "intensity": 0.0}}
  ],
  "point_lights": [
    {{"id": "<id from input list>", "on": true,
      "offset_m": [dx, dy, dz],
      "color": [r, g, b], "intensity": 0.0,
      "radius_m": 0.05, "falloff": "inverse_square",
      "rationale": "<one short phrase>"}}
  ],
  "windows": [
    {{"id": "<id from input list>", "daylight_on": true,
      "color": [r, g, b], "intensity": 0.0,
      "rationale": "<one short phrase>"}}
  ],
  "lights_only": true,
  "base_dim": 0.70,
  "grading_strength": 0.0,
  "grading_rationale": "<unused in lights_only mode>"
}}
"""


def _format_sources(sources: list[dict]) -> str:
    lines = []
    for s in sources:
        sid = f"{s['source']}_{s['kind']}_{s['index']:02d}"
        s["id"] = sid  # mutate in-place so emissive editor + render can join
        pos = ", ".join(f"{v:.2f}" for v in s["position_m"])
        size = s.get("size_m") or {}
        sz = (f"w={size.get('width_m', 0):.2f} "
              f"h={size.get('height_m', 0):.2f} "
              f"d={size.get('depth_m', 0):.2f}")
        wall = f"  wall={s['wall']}" if s.get("wall") else ""
        lines.append(f"  - id={sid}  kind={s['kind']}  phrase={s['phrase']!r}  "
                     f"pos=({pos})  size=[{sz}]{wall}")
    return "\n".join(lines) if lines else "  (none)"


def _format_room(out_dir: Path) -> str:
    fp = out_dir / "floorplan_analysis.json"
    if not fp.exists():
        return "  (floorplan_analysis.json missing)"
    import json
    try:
        room = json.loads(fp.read_text()).get("room", {})
        room_type = room.get("room_type", "room")
        dims = room.get("dimensions") or {}
        w = dims.get("width_m") or dims.get("width")
        d = dims.get("depth_m") or dims.get("depth")
        h = dims.get("height_m") or dims.get("height")
        return (f"  room_type={room_type}\n"
                f"  size_m: width={w} depth={d} height={h}")
    except Exception:
        return "  (floorplan_analysis.json unreadable)"


def estimate(output_dir: str | Path, image_path: str | Path,
             sources: list[dict]) -> dict:
    out_dir = Path(output_dir)
    blob = _format_sources(sources)
    prompt = _PROMPT.format(
        light_sources_blob=blob,
        room_blob=_format_room(out_dir),
    )

    ref = Image.open(str(image_path)).convert("RGB")
    # Downscale to keep the VLM payload small — lighting is a low-frequency cue.
    ref.thumbnail((1280, 1280), Image.LANCZOS)

    parsed = _vlm_call(prompt, [ref], max_tokens=2048)
    return _validate(parsed, sources)


def _validate(parsed: Any, sources: list[dict]) -> dict:
    """Fill in defaults so downstream code can rely on every key being present."""
    if not isinstance(parsed, dict):
        parsed = {}

    out: dict = {
        "ambient": parsed.get("ambient") or {"color": [1.0, 0.95, 0.88],
                                             "intensity": 0.15},
        "directional": parsed.get("directional") or [],
        "point_lights": [],
        "windows": [],
        "lights_only": bool(parsed.get("lights_only", True)),
        "base_dim": float(min(max(
            float(parsed.get("base_dim", 0.70)), 0.30), 1.00)),
        "grading_strength": float(min(max(
            float(parsed.get("grading_strength", 0.0)), 0.0), 0.80)),
        "grading_rationale": parsed.get("grading_rationale", ""),
    }
    # Clamp ambient intensity if the VLM returned something too hot.
    out["ambient"]["intensity"] = float(min(out["ambient"].get("intensity", 0.08), 0.30))
    # Clamp any directional intensities supplied by the VLM.
    for d in out["directional"]:
        if isinstance(d, dict):
            d["intensity"] = float(min(d.get("intensity", 0.0), 0.50))

    incoming_pl = {p.get("id"): p for p in (parsed.get("point_lights") or [])
                   if isinstance(p, dict)}
    incoming_w = {w.get("id"): w for w in (parsed.get("windows") or [])
                  if isinstance(w, dict)}

    for s in sources:
        sid = s["id"]
        if s["kind"] in ("lamp", "ceiling_fixture"):
            p = incoming_pl.get(sid) or {}
            size_h = (s.get("size_m") or {}).get("height_m", 0.2)
            on = bool(p.get("on", True))
            intensity = float(p.get("intensity", 0.15))
            # Hard cap so a hot VLM estimate can't blow out the preview.
            intensity = min(intensity, 0.50) if on else 0.0
            out["point_lights"].append({
                "id": sid,
                "on": on,
                "offset_m": list(p.get("offset_m", [0.0, 0.7 * size_h, 0.0])),
                "color": list(p.get("color", [1.0, 0.85, 0.6])),
                "intensity": intensity,
                "radius_m": float(p.get("radius_m", 0.05)),
                "falloff": p.get("falloff", "inverse_square"),
                "rationale": p.get("rationale", ""),
            })
        elif s["kind"] == "window":
            w = incoming_w.get(sid) or {}
            daylight_on = bool(w.get("daylight_on", True))
            intensity = float(w.get("intensity", 0.20))
            intensity = min(intensity, 0.60) if daylight_on else 0.0
            out["windows"].append({
                "id": sid,
                "daylight_on": daylight_on,
                "color": list(w.get("color", [1.0, 0.97, 0.92])),
                "intensity": intensity,
                "rationale": w.get("rationale", ""),
            })

    return out