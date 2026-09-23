"""Shared VLM backend router for object-placement modules.

All VLM chat calls across furniture / wall-mounted / decoration modules go
through ``vlm_post`` so the backend is switchable with one env var.  Every
supported backend speaks the OpenAI-compatible ``chat/completions`` API, so the
response JSON is identical and callers are unchanged.

You need exactly one of these — a local Qwen server, or a hosted GPT API:

    SCENEWEAVE_VLM_BACKEND = "qwen" (default) | "gpt"

``qwen``
    A local vLLM server (see README → Services).  No API key, no per-call cost,
    needs a GPU.

``gpt``
    Any OpenAI-compatible hosted endpoint.  No GPU needed.  Set a key and, if
    you are not using api.openai.com, the URL and model id:

        OPENAI_API_KEY            key (NVIDIA_API_KEY / SCENEWEAVE_GPT55_KEY
                                  are also accepted)
        SCENEWEAVE_VLM_MODEL      model id (default: gpt-5.5)
        SCENEWEAVE_VLM_URL        endpoint (default: api.openai.com)

``gpt55`` stays as an alias for ``gpt`` and keeps its original defaults —
NVIDIA's gateway with ``openai/openai/gpt-5.5`` — so existing setups that
export ``NVIDIA_API_KEY`` keep working untouched.

Env:
    SCENEWEAVE_VLM_BACKEND = "qwen" (default) | "gpt" | "gpt55"
    VLM_API_URL            = local Qwen endpoint (default localhost:8080)
"""
from __future__ import annotations

import os

_DEFAULT_QWEN_URL = "http://localhost:8080/v1/chat/completions"
_OPENAI_VLM_URL   = "https://api.openai.com/v1/chat/completions"
_NVIDIA_VLM_URL   = "https://inference-api.nvidia.com/v1/chat/completions"
_OPENAI_MODEL     = "gpt-5.5"
_GPT55_VLM_MODEL  = "openai/openai/gpt-5.5"     # NVIDIA gateway's id for it

_GPT_ALIASES = ("gpt", "openai", "gpt55", "gpt-5.5", "nvidia")


def backend() -> str:
    return os.environ.get("SCENEWEAVE_VLM_BACKEND", "qwen").strip().lower()


def _is_reasoning_model(model: str) -> bool:
    """gpt-5.x and the o-series reason before answering, which changes what the
    API accepts: they reject an explicit temperature/top_p, and their
    ``max_tokens`` budget covers reasoning as well as the visible answer."""
    m = model.rsplit("/", 1)[-1].lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def gpt_config() -> tuple[str | None, str, str]:
    """(key, url, model) for the hosted backend.

    The default endpoint follows the key that is set: an explicit
    ``NVIDIA_API_KEY`` (or the historical ``gpt55`` backend name) routes to
    NVIDIA's gateway, anything else goes to api.openai.com.  ``SCENEWEAVE_VLM_URL``
    and ``SCENEWEAVE_VLM_MODEL`` override both.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    nvidia_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("SCENEWEAVE_GPT55_KEY")
    via_nvidia = bool(nvidia_key) and (not openai_key
                                       or backend() in ("gpt55", "gpt-5.5", "nvidia"))
    key = nvidia_key if via_nvidia else (openai_key or nvidia_key)
    url = os.environ.get("SCENEWEAVE_VLM_URL") or (
        _NVIDIA_VLM_URL if via_nvidia else _OPENAI_VLM_URL)
    model = os.environ.get("SCENEWEAVE_VLM_MODEL") or (
        _GPT55_VLM_MODEL if via_nvidia else _OPENAI_MODEL)
    return key, url, model


def vlm_post(payload: dict, timeout: int = 90, qwen_url: str | None = None):
    """POST an OpenAI-style chat payload to the active VLM backend and return the
    raw requests.Response (so .json()/.raise_for_status()/.status_code work)."""
    import requests
    if backend() in _GPT_ALIASES:
        key, url, model = gpt_config()
        if key:
            p = dict(payload)
            p["model"] = model
            p.pop("chat_template_kwargs", None)   # Qwen-only; OpenAI API rejects it
            if _is_reasoning_model(model):
                # Reasoning models ONLY support the default temperature/top_p (1)
                # and return HTTP 400 on the temperature=0.0 the placement code
                # sends for determinism.  Drop them so the request is accepted.
                p.pop("temperature", None)
                p.pop("top_p", None)
                # max_tokens covers REASONING + output, so a Qwen-sized budget
                # (e.g. 2048) gets eaten by reasoning and returns empty content
                # on complex prompts.  Raise the floor.
                p["max_tokens"] = max(int(p.get("max_tokens", 1024)) + 6000, 8192)
            import time
            # Hosted endpoints intermittently return 504/502/429 under load. A
            # single POST then fails the whole call (and silently drops a
            # decoration). Retry transient errors with backoff so they recover.
            _r = None
            for _attempt in range(3):          # 1 try + 2 retries
                try:
                    _r = requests.post(url, json=p,
                                       headers={"Authorization": f"Bearer {key}"},
                                       timeout=timeout)
                except (requests.Timeout, requests.ConnectionError) as _e:
                    if _attempt < 2:
                        print(f"[vlm] {model} {type(_e).__name__} — retry {_attempt+1}/2")
                        time.sleep(3 * (_attempt + 1)); continue
                    raise
                if _r.status_code in (429, 500, 502, 503, 504) and _attempt < 2:
                    print(f"[vlm] {model} HTTP {_r.status_code} — retry {_attempt+1}/2 after backoff")
                    time.sleep(3 * (_attempt + 1)); continue
                break
            if _r.status_code != 200:
                # Surface the server-side reason (image_parse_error, bad param,
                # rate limit …) — callers only see "400 Client Error" otherwise.
                _nimg = sum(1 for m in p.get("messages", [])
                            for c in (m.get("content") or [])
                            if isinstance(c, dict) and c.get("type") == "image_url")
                print(f"[vlm] {model} HTTP {_r.status_code} (imgs={_nimg}): {_r.text[:300]}")
            return _r
        print(f"[vlm] SCENEWEAVE_VLM_BACKEND={backend()} but no API key "
              f"(set OPENAI_API_KEY or NVIDIA_API_KEY) — falling back to Qwen")
    return requests.post(qwen_url or os.environ.get("VLM_API_URL", _DEFAULT_QWEN_URL),
                         json=payload, timeout=timeout)
