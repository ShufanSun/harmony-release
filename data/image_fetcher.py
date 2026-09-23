"""
image_fetcher.py — fetch credit-free indoor-corner photos and VLM-filter.

Searches a stock-photo provider (Pexels by default; Pixabay / Unsplash also
supported) for the keyword "indoor scene room corner", downloads N candidates,
runs a focused VLM check on each one, and moves images that show a CLEAR room
interior with at least one deepest visible wall corner into
`data/fetched_data/VLM-filtered/`.  Rejected images stay in
`data/fetched_data/raw/` for inspection.

Usage:
    # With Pexels (recommended — free, attribution suggested but not required)
    export PEXELS_API_KEY="your_key"
    python data/image_fetcher.py

    # With Pixabay (also free)
    export PIXABAY_API_KEY="your_key"
    python data/image_fetcher.py --provider pixabay

    # With Unsplash
    export UNSPLASH_API_KEY="your_key"
    python data/image_fetcher.py --provider unsplash

    # Custom keyword / count / output dir
    python data/image_fetcher.py \
        --query "interior room design" \
        --count 30 \
        --output-root data/fetched_data

VLM:
    Uses the same VLM API URL the rest of the codebase uses (Qwen3 at
    http://localhost:8080/v1/chat/completions by default; override via the
    VLM_API_URL env var).

Notes on attribution:
    Pexels images are free for commercial / non-commercial use, no attribution
    REQUIRED but recommended.  Pixabay similar.  Unsplash similar.  Each
    downloaded image's metadata (photographer, source URL) is recorded in
    fetched_data/manifest.json so you can credit creators if you publish.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

VLM_API_URL = os.environ.get(
    "VLM_API_URL",
    "http://localhost:8080/v1/chat/completions",
)

# Diverse indoor room types (comma-separated). The CLI fetches from each in turn
# so the dataset isn't limited to living-room corners. Add/remove types freely.
_DEFAULT_QUERY = ("indoor kitchen,indoor balcony,sunroom interior,"
                  "dining room interior,home office interior,living room interior,"
                  "bedroom interior,bathroom interior,indoor room corner")
_DEFAULT_COUNT = 20
_DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "fetched_data"
_USER_AGENT = "HARMONY-image-fetcher/1.0"


# ── Provider clients ─────────────────────────────────────────────────────────


def _fetch_pexels(query: str, count: int, offset: int = 0) -> "list[dict]":
    """Pexels Search API.  Returns list of dicts with download URL + metadata.
    Caps at 80 per page; we paginate when count > 80.  `offset` is the
    number of search results to skip — used by the run() refill loop to
    request *new* images rather than the same first-N each time."""
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PEXELS_API_KEY env var not set.  Get a free key at "
            "https://www.pexels.com/api/")
    headers = {"Authorization": api_key, "User-Agent": _USER_AGENT}
    out: list[dict] = []
    per_page = min(count, 80)
    start_page = (offset // per_page) + 1
    pages_needed = (count + per_page - 1) // per_page
    for page in range(start_page, start_page + pages_needed):
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers, timeout=30,
            params={"query": query, "per_page": per_page,
                    "page": page, "orientation": "landscape"})
        r.raise_for_status()
        for p in r.json().get("photos", []):
            out.append({
                "provider":     "pexels",
                "id":           str(p.get("id")),
                "url_full":     p.get("src", {}).get("large2x")
                                or p.get("src", {}).get("large")
                                or p.get("src", {}).get("original"),
                "photographer": p.get("photographer", ""),
                "source_url":   p.get("url", ""),
                "width":        int(p.get("width", 0)),
                "height":       int(p.get("height", 0)),
            })
            if len(out) >= count:
                return out
    return out


def _fetch_pixabay(query: str, count: int, offset: int = 0) -> "list[dict]":
    api_key = os.environ.get("PIXABAY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PIXABAY_API_KEY env var not set.  Get a free key at "
            "https://pixabay.com/api/docs/")
    out: list[dict] = []
    per_page = min(count, 200)
    start_page = (offset // per_page) + 1
    pages_needed = (count + per_page - 1) // per_page
    for page in range(start_page, start_page + pages_needed):
        r = requests.get(
            "https://pixabay.com/api/", timeout=30,
            headers={"User-Agent": _USER_AGENT},
            params={"key": api_key, "q": query,
                    "per_page": per_page, "page": page,
                    "orientation": "horizontal", "image_type": "photo",
                    "category": "buildings"})
        r.raise_for_status()
        for h in r.json().get("hits", []):
            out.append({
                "provider":     "pixabay",
                "id":           str(h.get("id")),
                "url_full":     h.get("largeImageURL") or h.get("webformatURL"),
                "photographer": h.get("user", ""),
                "source_url":   h.get("pageURL", ""),
                "width":        int(h.get("imageWidth", 0)),
                "height":       int(h.get("imageHeight", 0)),
            })
            if len(out) >= count:
                return out
    return out


def _fetch_unsplash(query: str, count: int, offset: int = 0) -> "list[dict]":
    api_key = os.environ.get("UNSPLASH_API_KEY")
    if not api_key:
        raise RuntimeError(
            "UNSPLASH_API_KEY env var not set.  Get a free key at "
            "https://unsplash.com/developers")
    headers = {"Authorization": f"Client-ID {api_key}",
               "User-Agent":    _USER_AGENT}
    out: list[dict] = []
    per_page = min(count, 30)
    start_page = (offset // per_page) + 1
    pages_needed = (count + per_page - 1) // per_page
    for page in range(start_page, start_page + pages_needed):
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            headers=headers, timeout=30,
            params={"query": query, "per_page": per_page,
                    "page": page, "orientation": "landscape"})
        r.raise_for_status()
        for p in r.json().get("results", []):
            out.append({
                "provider":     "unsplash",
                "id":           str(p.get("id")),
                "url_full":     p.get("urls", {}).get("regular"),
                "photographer": p.get("user", {}).get("name", ""),
                "source_url":   p.get("links", {}).get("html", ""),
                "width":        int(p.get("width", 0)),
                "height":       int(p.get("height", 0)),
            })
            if len(out) >= count:
                return out
    return out


_PROVIDER_FNS = {
    "pexels":      _fetch_pexels,
    "pixabay":     _fetch_pixabay,
    "unsplash":    _fetch_unsplash,
}


# ── Download ─────────────────────────────────────────────────────────────────


def _download(url: str, dst: Path, timeout: int = 60) -> bool:
    if dst.exists():
        return True
    try:
        r = requests.get(url, stream=True, timeout=timeout,
                         headers={"User-Agent": _USER_AGENT})
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  [download] failed {url}: {e}")
        if dst.exists():
            try: dst.unlink()
            except Exception: pass
        return False


# ── VLM filter ───────────────────────────────────────────────────────────────


_VLM_PROMPT = """\
You are screening photographs for a 3D room-reconstruction dataset.
The reconstruction pipeline needs a CLEAR INDOOR scene with the
CEILING and the FLOOR visible.  We want VARIETY of indoor room types —
living rooms, bedrooms, KITCHENS, dining rooms, home offices, indoor
BALCONIES / sunrooms, bathrooms, hallways — not only living-room corners.

I am giving you ONE photograph.  Apply the conditions below.
Conditions (C), (G), and (H) are HARD requirements — no leniency
on those, even for borderline cases.  A visible room CORNER (B) is
PREFERRED but NOT required — frontal / single-wall views of kitchens,
balconies, and other indoor rooms are welcome.  For all other conditions,
when in doubt about a marginal case, ACCEPT.  We'd rather include
borderline images and filter later than throw away usable data.

BEFORE WRITING YOUR ANSWER, walk through each condition in order
and check it explicitly against pixels you can actually see.  If
you find yourself reasoning "the room probably has a ceiling out of
frame" or "there's likely a corner just outside the visible area",
you have failed the check — that means the feature is NOT in the
photograph.  Only what is literally pictured counts.

★ GEOMETRIC PRIORITIES — these are the main gate ★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  (B) ROOM CORNER — PREFERRED, NOT REQUIRED (do NOT reject for missing it)
      A visible vertical wall-meets-wall corner is helpful for
      reconstruction, but it is OPTIONAL now.  Do NOT reject an image
      just because no corner is visible: a fronto-parallel / single-wall
      view of a kitchen, balcony, bathroom or any other clear indoor room
      is ACCEPTABLE as long as the ceiling (C), floor (D), indoor (E),
      no-text (G) and simple box geometry (H) conditions hold.
      (If a corner IS visible it should be a real wall-meets-wall crease,
      not a door/furniture/window edge — but its absence is fine.)

      A single-wall / fronto-parallel view (no corner) is FINE — do not
      reject for it.  A corner is not required, so there is nothing to
      REJECT under (B).

  (C) CEILING VISIBLE — HARD, no leniency
      The ceiling of the room MUST be visible in the photograph.
      You MUST be able to see:
        • ceiling pixels (an actual ceiling surface) in the upper
          portion of the image, AND
        • the wall-ceiling junction at the top of the corner crease
          from (B).
      Sloped / cathedral / coffered ceilings all count.  A thin strip
      of ceiling along the top edge is enough — but it MUST be an
      identifiable ceiling surface, not inferred.

      ★ PROCEDURAL CHECK — apply this literally ★
      Look at the very TOP EDGE of the image, the topmost ~5 % strip.
      Ask: "what surface is in this strip?"
        • Mostly CEILING surface (a flat plane oriented horizontally,
          usually white/cream/beige, possibly with a recessed light or
          beam) → (C) PASSES.
        • Mostly WALL surface (the same vertical plane as the rest of
          the wall — paint, wallpaper, art frames, window frames,
          curtain rods touching the top) → (C) FAILS.  The ceiling is
          cropped out of frame.

      ★ STRONG HINTS that the ceiling is OUT OF FRAME (→ REJECT) ★
        ✗ A window frame, painting, or curtain rod reaches NEAR or AT
          the top edge of the image.
        ✗ Wall art / paintings extend almost to the top of the image
          — meaning there is wall surface above them, not ceiling.
        ✗ The image looks like a typical real-estate listing photo
          where the photographer cropped to fit the furniture; these
          almost always cut the ceiling out.
        ✗ A curtain rod / valance is the highest visible architectural
          element.

      REJECT — without leniency — if any of these is true:
        ✗ the photo's top edge cuts off before the ceiling appears
        ✗ the ceiling is fully hidden behind chandeliers / pendants /
          dropped fixtures spanning the entire top of the image
        ✗ you cannot identify where the wall ends and the ceiling
          starts (they blend, are out of frame, or are obscured)
        ✗ the corner crease from (B) runs off the top of the image
          before reaching a ceiling junction
      DO NOT accept "probably has a ceiling out of frame" — if the
      ceiling isn't actually pictured, (C) fails.

  (D) FLOOR VISIBLE
      You can see floor pixels somewhere in the lower portion of the
      image AND the bottom of the corner crease from (B) is in
      frame.  Furniture / a rug occluding most of the floor is fine
      as long as the wall-floor junction at the corner is readable.
      Reject only when the floor is genuinely cut off (e.g. only the
      upper half of the room is photographed).

  The CEILING (C) and FLOOR (D) are the required geometric anchors
  (a corner (B), when present, links them but is not required).  (C) is
  the strictest — no marginal acceptance.

★ ADDITIONAL CONDITIONS ★
━━━━━━━━━━━━━━━━━━━━━━━━

  (A) CLEAR VIEW OF AN INDOOR ROOM
      The photo shows the inside of a real room or a photoreal-render
      of a real-looking room — not a close-up of furniture, not an
      abstract empty-box CGI, not an exterior shot, not a tiny detail
      crop.  Photoreal CGI / interior renders are FINE as long as
      they look like an actual room someone might live in.

  (E) NORMAL INDOOR ARCHITECTURAL SPACE
      Be permissive: any normal room type (living room, bedroom,
      kitchen, dining room, office, hallway, bathroom, study, foyer,
      etc.) is fine.  Reject only obviously non-architectural cases:
        ✗ vehicle interiors  (car / RV / boat / cabin)
        ✗ industrial process spaces  (factory floor, server room,
          machine hall, warehouse with no finished walls)
        ✗ unfinished construction  (bare studs, raw concrete, gutted
          renovation, drywall not painted)
        ✗ covered patios / balconies / atriums whose "walls" are
          actually open columns

  (F) NOT EMPTY — at least one object on the floor
      The image must show at least ONE piece of furniture, plant,
      rug, or visible decor resting on the floor.  Wholly empty
      rooms (showroom shots, real-estate "before" photos with bare
      floors, empty staging renders) are REJECTED.  A single chair
      or one rug counts as "something" — the bar is just "non-empty".

  (G) NO WATERMARKS / TEXT OVERLAYS — HARD, no leniency
      REJECT immediately if you can see ANY of these on the image:
        ✗ a stock-photo brand watermark — these include but are not
          limited to: Shutterstock, Adobe Stock, Getty Images,
          iStock, Alamy, Dreamstime, LovePik, 123rf, Depositphotos,
          Vecteezy, Bigstock, Pikbest, Pexels (typed across image),
          Unsplash (typed across image), VectorStock, Stocksy
        ✗ a photographer / agency LOGO overlay (typed words AND/OR
          a graphic mark drawn onto the image)
        ✗ a URL or web address printed on the image
          (e.g. "lovepik.com", "www.alamy.com")
        ✗ a stock-image identifier code (e.g. "2DH6NTR", "ID 1234567")
          rendered onto the image
        ✗ "© [name]" copyright text, "Photo by …" attribution typed
          on top of the image
        ✗ semi-transparent diagonal text covering parts of the photo
        ✗ a coloured frame / border drawn around the photo (a coloured
          stripe along the edge that isn't part of the room itself)
      A tiny embedded photographer credit confined to one corner that
      occupies < 1 % of image area MAY be tolerated — but be strict:
      if the text is readable, REJECT.

  (H) SIMPLE BOX-LIKE (MANHATTAN) ROOM GEOMETRY — HARD, no leniency
      Our reconstruction pipeline assumes a Manhattan-world room: a
      SINGLE rectangular room with exactly 4 vertical walls meeting at
      right angles, a flat horizontal ceiling, and a flat horizontal
      floor.  We are looking for ONE orthographic corner of ONE simple
      box-shaped room — nothing else.  Anything more complicated than
      that produces bad reconstructions and MUST be rejected.

      ★ PROCEDURAL CHECK — apply this literally ★
      STEP 1: Count the rooms.  Scan the entire photo and ask: "how
              many distinct rooms are visible in this image?"  If the
              answer is more than one — for example, you can see a
              living room AND a kitchen, OR a hallway AND a room
              beyond, OR any second room visible through an open
              archway / wide opening / pass-through — REJECT (H).
              Only doors and windows that lead OUTSIDE or to a closed
              door are fine.  An open archway revealing another room's
              floor and walls is NOT fine.
      STEP 2: Check the floorplan footprint.  Mentally trace the
              outline of the floor.  Is it a simple rectangle?
                ✓ rectangle / square → PASS
                ✗ L-shape, T-shape, zig-zag, multi-zone, or a
                  rectangle that opens into another rectangle through
                  a wide opening → REJECT (H)
      STEP 3: Check the ceiling.  Is it a single flat horizontal
              plane?  Vaulted, sloped, A-frame, exposed beams forming
              the ceiling shape, or any ceiling that is NOT one flat
              plane → REJECT (H).
      STEP 4: Check the walls.  Are all visible walls flat vertical
              planes meeting at right angles?  Curved walls, angled
              accent walls, bay windows forming curved sections,
              glass-corner walls (two adjacent floor-to-ceiling glass
              walls with no vertical wall between them) → REJECT (H).
      STEP 5: Check for vertical complexity.  Is everything on ONE
              level?  Lofts, mezzanines, sunken seating areas, split
              levels, internal staircases, double-height atriums with
              an upper balcony visible → REJECT (H).

      ACCEPT typical rectangular rooms:
        ✓ standard living room / bedroom / kitchen / office with 4
          straight vertical walls meeting at right angles, fully
          enclosed (no second room visible)
        ✓ flat horizontal ceiling (drywall, plaster, drop ceiling
          with rectangular tiles, popcorn texture — all flat)
        ✓ minor architectural niceties (a single shallow recessed
          nook, crown moulding, baseboards) are fine
        ✓ closed doors, windows looking outside, mirrors on a wall

      REJECT — without leniency — if any of the following is true:
        ✗ MORE THAN ONE ROOM visible.  This is the most common failure.
          Examples that all FAIL:
            • a living room with a kitchen visible through an opening
            • a kitchen island/peninsula that opens directly into a
              living/dining area on the other side
            • a hallway shot where you can see into a room beyond
            • a dining area visible from the living room through a
              wide cased opening
            • an open-plan loft where multiple zones share the space
          If you can stand at the photographer's position and walk
          into a different room WITHOUT going through a closed door,
          REJECT.
        ✗ vaulted / cathedral / sloped / pitched / A-frame ceilings
        ✗ exposed angled roof beams / trusses defining the ceiling
          shape (rustic barn or cabin ceilings)
        ✗ L-shaped, T-shaped, U-shaped, or any non-rectangular
          floorplan
        ✗ glass-corner rooms, curved walls, round walls, octagonal
          rooms, bay windows forming a curved wall section
        ✗ multi-level interiors: lofts, mezzanines, sunken seating,
          split-level layouts, staircases inside the visible room
        ✗ atrium / double-height spaces with a balcony / upper level
          visible
        ✗ extreme architectural angles (slanted accent walls, wedge-
          shaped rooms)
        ✗ pony walls / half walls / partial dividers that create
          separate-but-connected zones within one space

      Skylights or recessed ceiling lights are fine — they're holes
      in an otherwise flat ceiling, not Manhattan violations.  A
      single closed door visible on a wall is fine — you cannot see
      the room beyond it, so it doesn't count as a second room.

★ SUMMARY OF REJECT CASES ★
  • single-wall flat-on shots with no visible corner crease (B)
  • photos that crop the ceiling out of frame (C)
  • exterior / outdoor shots
  • product / furniture catalog photography (one object on coloured
    background, no full room context)
  • abstract / artistic compositions, floor-plan diagrams, empty-box
    CGI / white-cube renders with no furnishings
  • tightly-cropped photos that hide the ceiling or the floor
  • the obviously non-architectural categories listed in (E)
  • completely empty rooms with bare floor (per F)
  • watermarked / branded / overlay-labelled images (per G)
  • non-Manhattan geometry: vaulted ceilings, L-shaped rooms,
    glass-corner rooms, lofts/mezzanines, curved walls (per H)

Return ONLY valid JSON, no markdown:
  {"accept": true,  "reasoning": "one short sentence noting what's good"}
  {"accept": false, "reasoning": "one short sentence noting which condition failed (A/B/C/D/E/F/G/H)"}
"""


def _encode_image_for_vlm(img_path: Path, max_side: int = 1024) -> "str | None":
    try:
        from PIL import Image
        im = Image.open(img_path).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"  [encode] failed {img_path}: {e}")
        return None


def _vlm_check(img_path: Path) -> "dict | None":
    b64 = _encode_image_for_vlm(img_path)
    if b64 is None:
        return None
    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _VLM_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        # Route through the shared VLM backend (gpt-5.5 gateway / Qwen) so the
        # check works without a local :8080 server. Falls back to the raw URL.
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from object_placement.vlm_backend import vlm_post as _vlm_post
            r = _vlm_post(payload, timeout=120)
        except Exception:
            r = requests.post(VLM_API_URL, json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print(f"  [vlm] no JSON in: {raw[:160]}")
            return None
        return json.loads(m.group())
    except Exception as e:
        print(f"  [vlm] call failed: {e}")
        return None


# ── Main pipeline ────────────────────────────────────────────────────────────


def run(query: str = _DEFAULT_QUERY,
        count: int = _DEFAULT_COUNT,
        output_root: "str | Path" = _DEFAULT_OUTPUT_ROOT,
        provider: "str | list[str]" = "pexels",
        skip_existing: bool = True,
        oversample_factor: float = 4.0,
        max_oversample: int = 200) -> dict:
    """Fetch images one-at-a-time, VLM-check each, KEEP only those that pass.
    Continue until `count` accepted images have been collected (or the
    candidate pool is exhausted, whichever comes first).

    Rejected images are deleted from disk — only the manifest entry is kept
    so re-runs don't re-process them.

    `oversample_factor` controls how big the initial candidate pool is
    relative to `count`.  A factor of 4 means we ask the provider for 80
    candidates when the user wants 20 accepted.  If we need more, we
    multiply (up to `max_oversample`).
    """
    output_root = Path(output_root)
    accepted_dir = output_root / "VLM-filtered"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Normalise `provider` to a chain: ["pexels", "pixabay", ...].  When the
    # current provider exhausts (refill returns 0 new candidates), the run
    # advances to the next.  This handles the common case of one provider
    # running out of unique results before we've hit the target count.
    if isinstance(provider, str):
        provider_chain = [p.strip() for p in provider.split(",") if p.strip()]
    else:
        provider_chain = list(provider)
    if not provider_chain:
        raise ValueError("provider chain is empty")
    for p in provider_chain:
        if p not in _PROVIDER_FNS:
            raise ValueError(f"Unknown provider {p!r}.  "
                             f"Available: {sorted(_PROVIDER_FNS)}")

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {"runs": [], "images": []}
    else:
        manifest = {"runs": [], "images": []}
    seen_ids: set = {(img.get("provider"), img.get("id"))
                     for img in manifest.get("images", [])}

    # Pre-existing accepted files in VLM-filtered/ count toward `count`.
    # The directory listing is the source of truth (the user may have
    # manually deleted some images).  Target is total-after-run, not
    # new-accepts-this-run.
    def _count_accepted_on_disk() -> int:
        return sum(1 for f in accepted_dir.iterdir() if f.is_file())

    starting_count = _count_accepted_on_disk()
    print(f"[fetch] {starting_count} image(s) already in {accepted_dir.relative_to(REPO_ROOT)}/  "
          f"— target after this run: {count}")
    if starting_count >= count:
        print(f"[fetch] Target {count} already met ({starting_count} on disk). "
              f"Nothing to do.  Use --no-skip-existing on a re-run, or raise "
              f"--count to add more.")
        return {"timestamp": datetime.now().isoformat(),
                "provider": provider_chain, "query": query, "target": count,
                "downloaded": 0, "accepted": 0, "rejected": 0,
                "fetch_calls": 0,
                "starting_count": starting_count}

    run_record = {
        "timestamp":      datetime.now().isoformat(),
        "provider_chain": provider_chain,
        "query":          query,
        "target":         count,
        "starting_count": starting_count,
        "downloaded":     0,
        "accepted":       0,
        "rejected":       0,
        "fetch_calls":    0,
        "providers_used": [],   # records which providers actually contributed
    }

    pool: list = []
    pool_size = max(int(count * oversample_factor), count + 5)
    fetched_so_far = 0

    # Per-provider state: search offset (so re-runs paginate forward),
    # plus an "exhausted" flag that gets set when a refill returns zero
    # new candidates — we don't keep banging on a dead provider.
    provider_state: dict = {p: {"offset": 0, "exhausted": False, "fetched": 0}
                            for p in provider_chain}
    current_idx = 0

    def _current_provider() -> "str | None":
        nonlocal current_idx
        while current_idx < len(provider_chain):
            p = provider_chain[current_idx]
            if not provider_state[p]["exhausted"]:
                return p
            current_idx += 1
        return None

    def _refill_pool(target_pool_size: int):
        nonlocal pool, fetched_so_far, current_idx
        ask = max(target_pool_size, 20)
        # Try providers in order until one yields new candidates or all are
        # exhausted.
        while True:
            cur = _current_provider()
            if cur is None:
                print(f"[fetch] All providers in chain {provider_chain} "
                      f"exhausted — no more candidates.")
                return
            st = provider_state[cur]
            print(f"[fetch] provider={cur}  query={query!r}  "
                  f"asking for {ask} candidates  (offset={st['offset']})")
            try:
                new_cands = _PROVIDER_FNS[cur](query, ask, offset=st["offset"])
            except TypeError:
                new_cands = _PROVIDER_FNS[cur](query, ask)
            except Exception as e:
                print(f"[fetch] {cur} error: {e}  → marking exhausted, advancing")
                st["exhausted"] = True
                current_idx += 1
                continue
            run_record["fetch_calls"] += 1
            st["offset"] += ask
            added = 0
            skipped_seen = 0
            for c in new_cands:
                key = (c.get("provider"), c.get("id"))
                if key in seen_ids:
                    skipped_seen += 1
                    continue
                if any(p.get("id") == c.get("id")
                       and p.get("provider") == c.get("provider")
                       for p in pool):
                    continue
                pool.append(c)
                added += 1
            fetched_so_far += len(new_cands)
            st["fetched"] += len(new_cands)
            print(f"[fetch] {cur}: +{added} new candidate(s)  "
                  f"(skipped {skipped_seen} already-seen, "
                  f"raw {len(new_cands)});  pool size now {len(pool)}  "
                  f"(total queried this run: {fetched_so_far})")
            if added > 0:
                if cur not in run_record["providers_used"]:
                    run_record["providers_used"].append(cur)
                return
            # No new candidates from this provider.  Two reasons it could
            # have happened: (a) the provider page is past the end of its
            # search results (returned 0 raw); (b) every result it returned
            # was already in the manifest.  Either way, mark this one
            # exhausted and move on to the next in the chain.
            print(f"[fetch] {cur} produced no new candidates — marking "
                  f"exhausted, advancing to next provider")
            st["exhausted"] = True
            current_idx += 1

    _refill_pool(pool_size)
    if not pool:
        print("[fetch] No candidates available — provider returned nothing.")
        return run_record

    idx = 0
    # Total = pre-existing accepted on disk + newly-accepted this run.
    # Keep going until the on-disk count reaches `count`.
    while (starting_count + run_record["accepted"]) < count:
        if not pool:
            if fetched_so_far >= max_oversample:
                print(f"[fetch] Hit max_oversample={max_oversample}; stopping "
                      f"with {starting_count + run_record['accepted']}/{count} "
                      f"on disk ({run_record['accepted']} new this run).")
                break
            print(f"[fetch] Pool exhausted, refilling…")
            _refill_pool(pool_size)
            if not pool:
                print(f"[fetch] No more candidates available; stopping with "
                      f"{starting_count + run_record['accepted']}/{count} "
                      f"on disk ({run_record['accepted']} new this run).")
                break

        c = pool.pop(0)
        idx += 1
        if not c.get("url_full"):
            continue

        ext = ".jpg"
        u = c["url_full"].split("?")[0].lower()
        for cand_ext in (".png", ".jpeg", ".jpg", ".webp"):
            if u.endswith(cand_ext):
                ext = cand_ext if cand_ext != ".jpeg" else ".jpg"
                break
        actual_provider = c.get("provider") or provider
        fname    = f"{actual_provider}_{c['id']}{ext}"
        raw_path = raw_dir / fname

        # Skip already-processed images (manifest hit) when skip_existing=True.
        already_logged = any(
            e.get("provider") == c.get("provider") and e.get("id") == c.get("id")
            for e in manifest.get("images", []))
        if skip_existing and already_logged:
            print(f"[try {idx}] {fname} — already processed, skipping")
            seen_ids.add((c.get("provider"), c.get("id")))
            continue

        # ── Download ─────────────────────────────────────────────────────────
        print(f"[try {idx}] downloading {fname} …")
        if not _download(c["url_full"], raw_path):
            continue
        run_record["downloaded"] += 1

        # ── VLM check ───────────────────────────────────────────────────────
        verdict  = _vlm_check(raw_path)
        accepted = bool(verdict and verdict.get("accept") is True)
        reason   = (verdict or {}).get("reasoning", "(no VLM response)")
        if accepted:
            run_record["accepted"] += 1
            dst = accepted_dir / fname
            try:
                shutil.move(raw_path, dst)   # MOVE — don't keep duplicate in raw/
                _on_disk = starting_count + run_record["accepted"]
                print(f"     ✓ accept ({_on_disk}/{count} on disk; "
                      f"+{run_record['accepted']} new this run) — "
                      f"{reason[:140]}")
                print(f"     saved → {dst.relative_to(REPO_ROOT)}")
            except Exception as e:
                print(f"     ✓ accept BUT move failed: {e}")
        else:
            run_record["rejected"] += 1
            print(f"     ✗ reject — {reason[:140]}")
            # Discard rejected images so disk doesn't fill up with junk.
            try:
                if raw_path.exists():
                    raw_path.unlink()
            except Exception:
                pass

        manifest["images"].append({
            "fname":        fname,
            "raw_path":     str(raw_path) if accepted is False else "",
            "accepted":     accepted,
            "reasoning":    reason,
            "provider":     actual_provider,
            "id":           c.get("id"),
            "photographer": c.get("photographer", ""),
            "source_url":   c.get("source_url", ""),
            "width":        c.get("width"),
            "height":       c.get("height"),
        })
        seen_ids.add((c.get("provider"), c.get("id")))
        time.sleep(0.3)

    manifest["runs"].append(run_record)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Cleanup: remove the now-empty raw/ staging dir contents.
    try:
        for f in raw_dir.iterdir():
            if f.is_file():
                f.unlink()
    except Exception:
        pass

    print()
    final_count = _count_accepted_on_disk()
    print("─" * 60)
    print(" FETCH SUMMARY")
    print("─" * 60)
    print(f"  query:           {query!r}")
    print(f"  provider:        {provider}")
    print(f"  target on disk:  {count}")
    print(f"  starting count:  {starting_count}")
    print(f"  new accepted:    {run_record['accepted']}")
    print(f"  final on disk:   {final_count}/{count}  → "
          f"{accepted_dir.relative_to(REPO_ROOT)}/")
    print(f"  fetch_calls:     {run_record['fetch_calls']}")
    print(f"  downloaded:      {run_record['downloaded']}")
    print(f"  rejected:        {run_record['rejected']}  (deleted from disk)")
    print(f"  manifest:        {manifest_path.relative_to(REPO_ROOT)}")
    if final_count < count:
        print(f"  ⚠ short of target by {count - final_count}; "
              f"re-run to keep adding (offset will advance, no re-fetch).")
    return run_record


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch credit-free indoor-corner photos and VLM-filter "
                    "for clear-view + deepest-corner visibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=_DEFAULT_QUERY,
                    help=f"Search keyword (default: {_DEFAULT_QUERY!r})")
    ap.add_argument("--count", type=int, default=_DEFAULT_COUNT,
                    help=f"Number of images to fetch (default {_DEFAULT_COUNT})")
    ap.add_argument("--output-root", default=str(_DEFAULT_OUTPUT_ROOT),
                    help=f"Root output dir (default {_DEFAULT_OUTPUT_ROOT})")
    ap.add_argument("--provider", default="pexels",
                    help=("Provider name(s).  Single: 'pexels'.  Comma-separated "
                          "chain (auto-fallback when one exhausts): "
                          "'pexels,pixabay,unsplash'.  Available: "
                          + ",".join(sorted(_PROVIDER_FNS))
                          + ".  All are free, credit-not-required stock-photo "
                            "APIs — no scraper-based providers (e.g. Pinterest, "
                            "Google/Bing image search) are included."))
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Re-process images already in the manifest "
                         "(default: skip ones we've already VLM-checked).")
    args = ap.parse_args()
    # --query may be a comma-separated list of scene types; fetch from each so the
    # dataset spans diverse indoor rooms (kitchen, balcony, …), not just corners.
    queries = [q.strip() for q in args.query.split(",") if q.strip()]
    per = max(1, -(-args.count // len(queries)))   # ceil-divide count across types
    total = {}
    for i, q in enumerate(queries):
        # Queries share one output dir, and run() counts accepted images already on
        # disk — so pass a CUMULATIVE target (each query adds `per` more) instead of
        # a flat `per` (which every query after the first sees as already met).
        cumulative = (i + 1) * per
        print(f"\n########## [{i+1}/{len(queries)}] query={q!r}  add {per} (cumulative target {cumulative}) ##########")
        res = run(query=q, count=cumulative,
                  output_root=args.output_root, provider=args.provider,
                  skip_existing=not args.no_skip_existing)
        if isinstance(res, dict):
            for k, v in res.items():
                total[k] = total.get(k, 0) + v if isinstance(v, (int, float)) else v
    print(f"\n########## ALL QUERIES DONE — {total} ##########")


if __name__ == "__main__":
    main()
