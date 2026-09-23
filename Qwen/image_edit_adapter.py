"""
image_edit_adapter.py — single entry point for image-edit requests, with a
runtime switch between the local Qwen image-edit server and Google's Gemini
image API.

Selection priority:
    1. Explicit `backend=` argument when calling `edit_image`
    2. `SCENEWEAVE_IMG_EDIT` env var  ("qwen" or "gemini")
    3. Default: "qwen"

Required env vars per backend:
    qwen   — none (assumes Qwen img_server running on localhost:8000)
    gemini — GEMINI_API_KEY  (and optionally SCENEWEAVE_GEMINI_MODEL,
                              default "gemini-2.5-flash-image")

The function accepts the existing Qwen `/generate` payload shape:
    {
        "prompt":              str,
        "negative_prompt":     str,           # optional
        "reference_image":     <b64 PNG>,     # optional
        "width":               int,           # qwen-only
        "height":              int,           # qwen-only
        "num_inference_steps": int,           # qwen-only
        "true_cfg_scale":      float,         # qwen-only
        "num_images":          int,           # qwen-only
    }

…and always returns the existing Qwen response shape:
    {"images": [<b64 PNG>, ...]}

When backend=gemini, the qwen-only fields (steps, cfg_scale, w/h, num_images)
are silently dropped.  The negative prompt is appended to the user prompt as
"AVOID: ..." since Gemini has no native negative-prompt field.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import requests

_QWEN_URL = os.environ.get("SCENEWEAVE_QWEN_IMG_EDIT_URL",
                            "http://localhost:8000/generate")
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-image"
# NVIDIA-hosted, OpenAI-compatible gateway. Gemini image models are served here
# under gateway-style ids (e.g. "gcp/google/gemini-3-pro-image-preview") and
# authenticate with an sk- NVIDIA_API_KEY — NOT a Google AIza key. A model id
# containing "/" routes here instead of Google's direct endpoint.
_NVIDIA_IMG_URL = "https://inference-api.nvidia.com/v1/chat/completions"


def _resolve_backend(override: Optional[str] = None) -> str:
    val = (override or os.environ.get("SCENEWEAVE_IMG_EDIT") or "qwen").lower().strip()
    if val not in ("qwen", "gemini"):
        print(f"[img_edit_adapter] unknown backend {val!r}, "
              f"defaulting to qwen", file=sys.stderr)
        return "qwen"
    return val


def edit_image(payload: dict, backend: Optional[str] = None,
               timeout: int = 300) -> dict:
    """Image-edit request → response, dispatched to the configured backend.

    See module docstring for payload / response shape.
    """
    chosen = _resolve_backend(backend)
    if chosen == "gemini":
        return _edit_via_gemini(payload, timeout=timeout)
    return _edit_via_qwen(payload, timeout=timeout)


def _edit_via_qwen(payload: dict, timeout: int = 300) -> dict:
    """Forward the payload as-is to the Qwen img_server /generate endpoint."""
    resp = requests.post(_QWEN_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _edit_via_gemini(payload: dict, timeout: int = 300) -> dict:
    """Translate Qwen-shaped payload to Gemini's contents/parts format,
    extract the returned image, return Qwen-shaped response."""
    model = os.environ.get("SCENEWEAVE_GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
    # Gateway-style model ids (e.g. gcp/google/gemini-3-pro-image-preview) are
    # served by the NVIDIA gateway with an sk- NVIDIA_API_KEY, not Google direct.
    if "/" in model:
        return _edit_via_gemini_gateway(payload, model, timeout=timeout)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY env var not set.  Set it, or switch back to "
            "Qwen with `export SCENEWEAVE_IMG_EDIT=qwen` (or unset)."
        )
    model = os.environ.get("SCENEWEAVE_GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)

    prompt_text = str(payload.get("prompt", "") or "")
    neg = str(payload.get("negative_prompt", "") or "")
    if neg:
        prompt_text = f"{prompt_text}\n\nAVOID the following: {neg}"

    parts: list[dict] = [{"text": prompt_text}]
    ref = payload.get("reference_image")
    if ref:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data":      ref,    # already base64-encoded
            }
        })

    body = {
        "contents":         [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    url = _GEMINI_ENDPOINT.format(model=model) + f"?key={api_key}"

    # Gemini image models intermittently return finishReason=NO_IMAGE (text-only
    # response, no picture) on busy/ambiguous inputs — a per-call flake, not a
    # hard error.  Retry a few times before giving up.
    tries = int(os.environ.get("SCENEWEAVE_GEMINI_NOIMAGE_RETRIES", "3"))
    data = None
    for attempt in range(max(1, tries)):
        resp = requests.post(url, json=body, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(
                f"Gemini HTTP {resp.status_code}: {resp.text[:600]}"
            )
        data = resp.json()
        # Walk candidates → content → parts; return the first inline_data image
        for cand in data.get("candidates", []) or []:
            for part in ((cand.get("content") or {}).get("parts") or []):
                for k in ("inline_data", "inlineData"):
                    blob = (part.get(k) or {}).get("data")
                    if blob:
                        return {"images": [blob]}
        # no image this attempt — retry
    raise RuntimeError(
        f"No image in Gemini response after {max(1, tries)} tries "
        f"(model {model!r}). First 400 chars: {str(data)[:400]}"
    )


def _edit_via_gemini_gateway(payload: dict, model: str, timeout: int = 300) -> dict:
    """Gemini image edit via the NVIDIA OpenAI-compatible gateway. The edited
    image comes back at choices[0].message.images[0].image_url.url as a data URI."""
    # Dedicated gateway-image key first, so the Gemini image key can differ from
    # the gpt-5.5 VLM key (they otherwise share NVIDIA_API_KEY).
    key = (os.environ.get("SCENEWEAVE_GEMINI_GATEWAY_KEY")
           or os.environ.get("NVIDIA_API_KEY") or os.environ.get("SCENEWEAVE_GPT55_KEY"))
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set for gateway Gemini image edit (model "
            f"{model!r}). Set NVIDIA_API_KEY, or use a Google model id + "
            "GEMINI_API_KEY, or SCENEWEAVE_IMG_EDIT=qwen."
        )
    prompt_text = str(payload.get("prompt", "") or "")
    neg = str(payload.get("negative_prompt", "") or "")
    if neg:
        prompt_text = f"{prompt_text}\n\nAVOID the following: {neg}"

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    ref = payload.get("reference_image")
    if ref:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + ref}})

    body = {"model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8192}
    # Gemini image models intermittently return a text-only completion (no image)
    # on busy/ambiguous inputs — retry a few times before failing.
    tries = int(os.environ.get("SCENEWEAVE_GEMINI_NOIMAGE_RETRIES", "3"))
    data = None
    for attempt in range(max(1, tries)):
        resp = requests.post(_NVIDIA_IMG_URL, json=body,
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"},
                             timeout=timeout)
        if not resp.ok:
            raise RuntimeError(f"Gemini(gateway) HTTP {resp.status_code}: {resp.text[:600]}")
        data = resp.json()
        for ch in data.get("choices", []) or []:
            for im in ((ch.get("message") or {}).get("images") or []):
                url = (im.get("image_url") or {}).get("url", "") or ""
                if "base64," in url:
                    return {"images": [url.split("base64,", 1)[1]]}
        # no image this attempt — retry
    raise RuntimeError(
        f"No image in gateway Gemini response after {max(1, tries)} tries "
        f"(model {model!r}). First 400 chars: {str(data)[:400]}"
    )


    return info
