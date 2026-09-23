"""Per-scene side-by-side comparison: input photo + 3 method renders.

For each scene in the 3-method intersection, produces a single panel image
with 4 sub-panels labeled `input`, `Harmony`, `3DREGEN`, `Gen3DSR`. Saved to
``evaluation/comparison_figures/<scene>.jpg``.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pillow_avif
except ImportError:
    pass
from PIL import Image

from loaders import load_3dregen, load_flat, load_gen3dsr, load_harmony

_HERE = os.path.dirname(os.path.abspath(__file__))

# Dataset / baseline paths: override via env vars (evaluation.sh exports them)
# or replace the placeholders below.
REF_DIR = os.environ.get("REF_DIR", "/path/to/testing_first1")
REGEN_DIR = os.environ.get("REGEN_DIR", "/path/to/3D-RE-GEN/results_batch")
GEN3DSR_DIR = os.environ.get("GEN3DSR_DIR", "/path/to/Gen3DSR/out/testing_first1")
# Comma-separated roots; earlier roots take precedence.
HARMONY_ROOTS = os.environ.get(
    "HARMONY_ROOTS",
    "/path/to/harmony/outputs/Demo/Artifact,/path/to/harmony/outputs/Demo/_processed",
)
OUT_DIR = os.path.join(_HERE, "comparison_figures")

PANEL_W = 640                    # width per sub-panel
GAP = 8                          # px between panels
LABEL_H = 0                      # no label band
BG = (255, 255, 255)             # white background between panels


def fit_panel(path: str, w: int, h: int) -> Image.Image:
    """Letterbox an image into a w x h panel."""
    img = Image.open(path).convert("RGB")
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), BG)
    canvas.paste(img, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def main():
    ref = load_flat(REF_DIR)
    har = load_harmony(HARMONY_ROOTS)
    reg = load_3dregen(REGEN_DIR)
    g3d = load_gen3dsr(GEN3DSR_DIR)

    shared = sorted(set(ref) & set(har) & set(reg) & set(g3d))
    print(f"3-method intersection: {len(shared)} scenes")

    os.makedirs(OUT_DIR, exist_ok=True)

    methods = [
        ("input",    ref),
        ("Harmony",  har),
        ("3DREGEN",  reg),
        ("Gen3DSR",  g3d),
    ]

    for i, scene in enumerate(shared, 1):
        # Determine per-scene aspect ratio from the input photo
        try:
            iw, ih = Image.open(ref[scene]).size
        except Exception:
            iw, ih = 800, 600
        panel_h = int(round(PANEL_W * (ih / iw)))
        total_w = PANEL_W * len(methods) + GAP * (len(methods) - 1)
        canvas = Image.new("RGB", (total_w, panel_h), BG)

        for col, (_, idx) in enumerate(methods):
            try:
                body = fit_panel(idx[scene], PANEL_W, panel_h)
                canvas.paste(body, (col * (PANEL_W + GAP), 0))
            except Exception:
                pass

        out = os.path.join(OUT_DIR, f"{scene}.jpg")
        canvas.save(out, quality=90)
        print(f"[{i}/{len(shared)}] {scene} → {out}")

    # Also build a contact-sheet index page (HTML) for quick browsing
    html = ["<html><head><style>",
            "body{background:#1a1a1a;color:#eee;font:14px sans-serif;margin:20px}",
            "h2{margin-top:32px;color:#bbb}",
            "img{max-width:100%;border:1px solid #333;margin-bottom:8px}",
            "</style></head><body>",
            f"<h1>Per-scene comparison ({len(shared)} scenes)</h1>",
            "<p>columns: input · Harmony · 3DREGEN · Gen3DSR</p>"]
    for scene in shared:
        html.append(f'<h2>{scene}</h2>')
        html.append(f'<img src="{scene}.jpg" alt="{scene}">')
    html.append("</body></html>")
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write("\n".join(html))
    print(f"\nbrowse: {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
