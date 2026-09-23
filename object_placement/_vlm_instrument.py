"""Lightweight VLM-call instrumentation (runtime analysis / rebuttal stats).

Monkeypatches requests.post to log every VLM chat-completions call (gpt55 or
Qwen) — timestamp, latency, #images — to the JSONL at SCENEWEAVE_VLM_LOG.
No-op unless SCENEWEAVE_VLM_LOG is set, so it's safe to import unconditionally.
Catches all call sites (vlm_post + direct VLM_API_URL posts) at one point.
"""
import os, time, json, threading
import requests

_LOG = os.environ.get("SCENEWEAVE_VLM_LOG")
if _LOG and not getattr(requests, "_vlm_patched", False):
    _orig_post = requests.post
    _lock = threading.Lock()

    def _count_images(kwargs):
        try:
            msgs = (kwargs.get("json") or {}).get("messages", [])
            n = 0
            for m in msgs:
                c = m.get("content")
                if isinstance(c, list):
                    n += sum(1 for part in c if isinstance(part, dict)
                             and part.get("type") == "image_url")
            return n
        except Exception:
            return -1

    def _patched(url, *args, **kwargs):
        is_vlm = isinstance(url, str) and "chat/completions" in url
        t0 = time.time()
        try:
            r = _orig_post(url, *args, **kwargs)
            ok = getattr(r, "status_code", 0) == 200
        finally:
            if is_vlm:
                rec = {"t0": t0, "dt": round(time.time() - t0, 3),
                       "stage": os.environ.get("SCENEWEAVE_STAGE", "?"),
                       "imgs": _count_images(kwargs),
                       "ok": ok if "ok" in dir() else None}
                with _lock:
                    with open(_LOG, "a") as f:
                        f.write(json.dumps(rec) + "\n")
        return r

    requests.post = _patched
    requests._vlm_patched = True
    print(f"[vlm-instrument] logging VLM calls → {_LOG}")