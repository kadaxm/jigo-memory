"""
Local voice-cloning TTS server (XTTS-v2, runs in the xtts_env side environment).

- Loads XTTS-v2 lazily on first use (~1.8GB download on first run ever)
- IDLE-UNLOAD: model unloads automatically after JIGO_UNLOAD_SECONDS (default
  600s) without a request — the idle server stays at a ~300MB skeleton instead
  of ~2.5-3GB resident. First request after an unload reloads (~85s).
- GPU (CUDA) if available, CPU fallback; auto-falls back to CPU on CUDA OOM
- If voice_sample.wav exists in the project root, speaks in the CLONED voice
- Otherwise falls back to a built-in studio speaker (install verification)
- POST /tts {text, language} -> audio/wav (24kHz) + X-Device / X-Synthesis-Ms headers
- POST /tts_stream_clone -> sentence-chunked raw PCM streaming
- GET /health -> {status, model_loaded, cloned, device, idle_unload_s}

Run (from project root):  xtts_env\\Scripts\\python.exe voice_server.py
"""

import gc
import io
import os
import re
import threading
import time
import wave

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

os.environ.setdefault("COQUI_TOS_AGREED", "1")

PROJECT = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(PROJECT, "voice_sample.wav")
IDLE_UNLOAD_SECONDS = int(os.getenv("JIGO_UNLOAD_SECONDS", "600"))

app = FastAPI(title="Jigo local voice server")
_tts = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_lock = threading.Lock()
_last_use = 0.0


def _engine():
    global _tts, _device, _last_use
    with _lock:
        _last_use = time.time()
        if _tts is None:
            from TTS.api import TTS
            try:
                _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device)
            except Exception as e:
                print(f"[GPU load failed ({e}) -> falling back to CPU]")
                _device = "cpu"
                _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
            print(f"[XTTS loaded on {_device} — will unload after {IDLE_UNLOAD_SECONDS}s idle]")
    return _tts


def _unload_if_idle():
    """Daemon: drop the model after IDLE_UNLOAD_SECONDS without a request.
    Lock-guarded, so it can never unload mid-synthesis."""
    global _tts
    while True:
        time.sleep(60)
        if _tts is None or time.time() - _last_use < IDLE_UNLOAD_SECONDS:
            continue
        with _lock:
            if _tts is None or time.time() - _last_use < IDLE_UNLOAD_SECONDS:
                continue
            _tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[XTTS unloaded — idle > {IDLE_UNLOAD_SECONDS}s, RAM freed]")


class Req(BaseModel):
    text: str
    language: str = "en"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _tts is not None,
        "cloned": os.path.exists(REFERENCE),
        "device": _device,
        "idle_unload_s": IDLE_UNLOAD_SECONDS,
    }


def _synth(text, language, cloned):
    """Synthesize on the active device; on CUDA OOM, move to CPU and retry."""
    global _last_use
    engine = _engine()
    with _lock:
        _last_use = time.time()
        if cloned:
            return engine.tts(text=text, speaker_wav=REFERENCE, language=language)
        return engine.tts(text=text, speaker=engine.speakers[0], language=language)


@app.post("/tts")
def tts(req: Req):
    global _tts, _device
    text = req.text.strip()[:600]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    cloned = os.path.exists(REFERENCE)
    t0 = time.perf_counter()
    try:
        try:
            wav = _synth(text, req.language, cloned)
        except torch.cuda.OutOfMemoryError:
            print("[CUDA OOM during synthesis -> switching to CPU for this and future requests]")
            with _lock:
                _tts = None
                _device = "cpu"
                torch.cuda.empty_cache()
            from TTS.api import TTS
            with _lock:
                _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
            wav = _synth(text, req.language, cloned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
    ms = round((time.perf_counter() - t0) * 1000)
    arr = np.array(wav, dtype=np.float32)
    pcm = (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Voice": "cloned" if cloned else "studio",
            "X-Device": _device,
            "X-Synthesis-Ms": str(ms),
        },
    )


def _warmup():
    """Touch the model once so the first real request isn't slow.
    OPT-IN ONLY (JIGO_WARMUP=1) — auto warm-up defeats idle-unload savings."""
    try:
        t0 = time.perf_counter()
        _synth("Hello.", "en", os.path.exists(REFERENCE))
        print(f"[warm-up done on {_device} in {time.perf_counter() - t0:.1f}s]")
    except Exception as e:
        print(f"[warm-up skipped: {e}]")


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text):
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


@app.post("/tts_stream_clone")
def tts_stream_clone(req: Req):
    """Sentence-chunked streaming: raw 16-bit PCM @24kHz, first sentence starts
    playing while the rest synthesize. On CUDA OOM mid-stream, switches to CPU."""
    global _tts, _device
    text = req.text.strip()[:600]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    cloned = os.path.exists(REFERENCE)
    sentences = _split_sentences(text)

    def gen():
        for sent in sentences:
            try:
                try:
                    wav = _synth(sent, req.language, cloned)
                except torch.cuda.OutOfMemoryError:
                    print("[CUDA OOM mid-stream -> switching to CPU]")
                    with _lock:
                        _tts = None
                        _device = "cpu"
                        torch.cuda.empty_cache()
                    from TTS.api import TTS
                    with _lock:
                        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
                    wav = _synth(sent, req.language, cloned)
            except Exception as e:
                print(f"[sentence synth failed, skipping: {e}]")
                continue
            arr = np.array(wav, dtype=np.float32)
            yield (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={"X-Voice": "cloned" if cloned else "studio"},
    )


if __name__ == "__main__":
    import uvicorn
    print(f"Jigo local voice server -> http://127.0.0.1:8100 (device: {_device})")
    print(f"XTTS loads on first use; idle-unload after {IDLE_UNLOAD_SECONDS}s (set JIGO_UNLOAD_SECONDS to change)")
    threading.Thread(target=_unload_if_idle, daemon=True).start()
    if os.getenv("JIGO_WARMUP") == "1":
        threading.Thread(target=_warmup, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8100)
