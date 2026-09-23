"""Thin wrapper around the qwen3 VLM endpoint used by the rest of HARMONY."""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image

VLM_API_URL = "http://localhost:8080/v1/chat/completions"


def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _first_json_block(text: str) -> str | None:
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group() if m else None


def call(prompt: str, images: list[Image.Image | Path | str],
         *, max_tokens: int = 1024, timeout: int = 180) -> dict | None:
    """Send `prompt` + `images` to the VLM. Returns the parsed JSON dict or None."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        if isinstance(img, (str, Path)):
            img = Image.open(str(img)).convert("RGB")
        b64 = _encode_png(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        # Route through the shared VLM backend so SCENEWEAVE_VLM_BACKEND=gpt55
        # (NVIDIA-hosted) is honoured like the rest of the pipeline, instead of
        # the hardcoded local Qwen server.  Falls back to the local URL if the
        # backend router is unavailable.
        try:
            from object_placement.vlm_backend import vlm_post as _vlm_post
            resp = _vlm_post(payload, timeout=timeout)
        except Exception:
            resp = requests.post(VLM_API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[lighting/vlm] request failed: {e}")
        return None

    block = _first_json_block(raw)
    if block is None:
        print(f"[lighting/vlm] no JSON in response: {raw[:200]}")
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError as e:
        print(f"[lighting/vlm] JSON parse failed: {e}\n{block[:300]}")
        return None
