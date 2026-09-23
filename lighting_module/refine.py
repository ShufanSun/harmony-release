"""VLM iterative refinement of the lighting setup.

Each iteration:
  1. Render a lit preview with the current `lights` dict.
  2. Show the VLM the reference photo + the lit preview side-by-side.
  3. Ask for delta updates: per-light intensity multiplier, colour shift,
     turning a light on/off, and ambient/directional tweaks.
  4. Apply the deltas, save the new `lights.json`, repeat.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .vlm_client import call as _vlm_call

_REFINE_PROMPT = """\
You are tuning a 3D lighting setup so that a rendered preview matches a \
reference photograph.

IMAGE 1 — REFERENCE PHOTO (the target appearance).
IMAGE 2 — CURRENT LIT PREVIEW (the current rendering with the lights below).

CURRENT LIGHTING SETUP (JSON):
{lights_blob}

LIGHT SOURCES IN THE SCENE (id, kind, position):
{sources_blob}

The renderer is in LIGHTS-ONLY mode: the base composite is preserved \
exactly, and each active light source ONLY adds a localised additive \
glow at its projected position (lamps → bulb halo, windows → halo over \
the window). There is no global wash, no ambient lift, no tone grade. So \
the only way to make a region brighter is to raise that light's \
intensity; the only way to dim is to lower it (or turn it off).

STEP 1 — DIAGNOSE THE PREVIEW vs THE REFERENCE. Pick:

  * `brightness_assessment` ∈ ["too_bright", "matches", "too_dim"]:
      - "too_bright"  preview's lamp/window halos are larger or hotter \
        than the reference suggests, or the highlight regions wash out \
        the underlying scene texture.
      - "too_dim"     preview's lamp/window highlights are barely visible \
        compared to the reference's bright spots.
      - "matches"     localised highlights already look right.
  * `tone_assessment` ∈ ["too_warm", "matches", "too_cool"]:
      describes the colour of the highlight regions.

STEP 2 — TRANSLATE INTO DELTAS. The simplest knob is \
`global_intensity_mult`, which scales EVERY active light's intensity at \
once by a single multiplier in [0.5, 1.3]:

  - too_bright → `global_intensity_mult` ≈ 0.6–0.8
  - too_dim    → `global_intensity_mult` ≈ 1.2
  - matches    → leave it 1.0

STEP 3 — fine-tune individual lights when a specific lamp/window is \
off-balance:

  - per-light intensity_mult MUST be in [0.5, 1.3]
  - colour shifts within ±0.05 per channel
  - never raise an absolute intensity above 0.5
  - if a light is clearly OFF in the reference, set `set_on=false` (or \
    `set_daylight_on=false` for windows)
  - if everything looks right, return empty arrays + verdict "converged"

STEP 4 — legacy global knobs (only used if `lights_only=false`):

  - `base_dim` ∈ [0.30, 1.00] — leave null to keep current
  - `grading_strength` ∈ [0, 0.8] — leave null to keep current

OUTPUT — strictly the following JSON object, no markdown:

{{
  "brightness_assessment": "matches",
  "tone_assessment": "matches",
  "global_intensity_mult": 1.0,
  "base_dim": null,
  "grading_strength": null,
  "ambient_delta": {{"color_delta": [dr, dg, db], "intensity_mult": 1.0}},
  "directional_delta": [
    {{"index": 0, "color_delta": [dr, dg, db], "intensity_mult": 1.0}}
  ],
  "point_light_deltas": [
    {{"id": "<id>", "set_on": true, "color_delta": [dr, dg, db],
      "intensity_mult": 1.0, "offset_delta_m": [dx, dy, dz]}}
  ],
  "window_deltas": [
    {{"id": "<id>", "set_daylight_on": true,
      "color_delta": [dr, dg, db], "intensity_mult": 1.0}}
  ],
  "verdict": "<one short sentence: closer | further | converged>"
}}
"""


def _format_sources_short(sources: list[dict]) -> str:
    parts = []
    for s in sources:
        if "id" not in s:
            continue
        pos = ", ".join(f"{v:.2f}" for v in s["position_m"])
        parts.append(f"  - {s['id']}  kind={s['kind']}  pos=({pos})")
    return "\n".join(parts) if parts else "  (none)"


def _clip_color(c: list[float]) -> list[float]:
    return [max(0.0, min(1.0, float(v))) for v in c]


def _apply_color_delta(base: list[float], delta) -> list[float]:
    if not isinstance(delta, (list, tuple)) or len(delta) != 3:
        return list(base)
    return _clip_color([float(b) + float(d) for b, d in zip(base, delta)])


def _apply_intensity_mult(base: float, mult, *, lo: float = 0.0,
                          hi: float = 0.5) -> float:
    try:
        m = float(mult)
    except Exception:
        return float(base)
    m = max(0.5, min(1.3, m))
    return max(lo, min(hi, float(base) * m))


def _apply_offset_delta(base: list[float], delta) -> list[float]:
    if not isinstance(delta, (list, tuple)) or len(delta) != 3:
        return list(base)
    return [float(b) + max(-0.2, min(0.2, float(d)))
            for b, d in zip(base, delta)]


def apply_deltas(lights: dict, deltas: dict) -> dict:
    """Return a new lights dict with `deltas` applied (clipped to safe ranges)."""
    new = {
        "ambient": dict(lights.get("ambient", {})),
        "directional": [dict(d) for d in lights.get("directional", [])],
        "point_lights": [dict(p) for p in lights.get("point_lights", [])],
        "windows": [dict(w) for w in lights.get("windows", [])],
        "base_dim": float(lights.get("base_dim", 0.70)),
        "grading_strength": float(lights.get("grading_strength", 0.30)),
        "grading_rationale": lights.get("grading_rationale", ""),
    }

    # Global one-shot scalar applied to EVERY active light's intensity. The
    # VLM uses this when its diagnosis is "scene too bright/dim overall" —
    # one knob is faster than per-light deltas. Clamped to [0.5, 1.3] per
    # iteration; the per-light hard caps below still apply afterwards.
    g_mult_raw = deltas.get("global_intensity_mult", 1.0)
    try:
        g_mult = float(g_mult_raw) if g_mult_raw is not None else 1.0
    except (TypeError, ValueError):
        g_mult = 1.0
    g_mult = max(0.5, min(1.3, g_mult))
    if abs(g_mult - 1.0) > 1e-3:
        new["ambient"]["intensity"] = float(min(
            float(new["ambient"].get("intensity", 0.0)) * g_mult, 0.30))
        for d in new["directional"]:
            d["intensity"] = float(min(float(d.get("intensity", 0.0)) * g_mult, 0.60))
        for p in new["point_lights"]:
            p["intensity"] = float(min(float(p.get("intensity", 0.0)) * g_mult, 0.50))
        for w in new["windows"]:
            w["intensity"] = float(min(float(w.get("intensity", 0.0)) * g_mult, 0.60))

    new_grading = deltas.get("grading_strength")
    if new_grading is not None:
        try:
            new["grading_strength"] = float(min(max(float(new_grading), 0.0), 0.80))
        except (TypeError, ValueError):
            pass

    new_base_dim = deltas.get("base_dim")
    if new_base_dim is not None:
        try:
            new["base_dim"] = float(min(max(float(new_base_dim), 0.30), 1.00))
        except (TypeError, ValueError):
            pass

    amb_d = deltas.get("ambient_delta") or {}
    new["ambient"]["color"] = _apply_color_delta(
        new["ambient"].get("color", [1, 1, 1]),
        amb_d.get("color_delta", [0, 0, 0]))
    new["ambient"]["intensity"] = _apply_intensity_mult(
        new["ambient"].get("intensity", 0.0),
        amb_d.get("intensity_mult", 1.0), hi=0.30)

    for d in deltas.get("directional_delta") or []:
        idx = int(d.get("index", -1))
        if 0 <= idx < len(new["directional"]):
            tgt = new["directional"][idx]
            tgt["color"] = _apply_color_delta(tgt.get("color", [1, 1, 1]),
                                              d.get("color_delta", [0, 0, 0]))
            tgt["intensity"] = _apply_intensity_mult(
                tgt.get("intensity", 0.0), d.get("intensity_mult", 1.0),
                hi=0.60)

    pl_index = {p["id"]: p for p in new["point_lights"]}
    for d in deltas.get("point_light_deltas") or []:
        tgt = pl_index.get(d.get("id"))
        if tgt is None:
            continue
        if "set_on" in d:
            tgt["on"] = bool(d["set_on"])
        tgt["color"] = _apply_color_delta(tgt.get("color", [1, 1, 1]),
                                          d.get("color_delta", [0, 0, 0]))
        tgt["intensity"] = _apply_intensity_mult(
            tgt.get("intensity", 0.0), d.get("intensity_mult", 1.0), hi=0.50)
        if "offset_delta_m" in d:
            tgt["offset_m"] = _apply_offset_delta(
                tgt.get("offset_m", [0, 0, 0]), d["offset_delta_m"])

    w_index = {w["id"]: w for w in new["windows"]}
    for d in deltas.get("window_deltas") or []:
        tgt = w_index.get(d.get("id"))
        if tgt is None:
            continue
        if "set_daylight_on" in d:
            tgt["daylight_on"] = bool(d["set_daylight_on"])
        tgt["color"] = _apply_color_delta(tgt.get("color", [1, 1, 1]),
                                          d.get("color_delta", [0, 0, 0]))
        tgt["intensity"] = _apply_intensity_mult(
            tgt.get("intensity", 0.0), d.get("intensity_mult", 1.0), hi=0.60)

    return new


def request_deltas(reference_image: Path, preview_image: Path,
                   lights: dict, sources: list[dict]) -> dict | None:
    import json as _json
    prompt = _REFINE_PROMPT.format(
        lights_blob=_json.dumps(lights, indent=2),
        sources_blob=_format_sources_short(sources),
    )
    ref = Image.open(str(reference_image)).convert("RGB")
    ref.thumbnail((1280, 1280), Image.LANCZOS)
    prev = Image.open(str(preview_image)).convert("RGB")
    prev.thumbnail((1280, 1280), Image.LANCZOS)
    return _vlm_call(prompt, [ref, prev], max_tokens=1536)
