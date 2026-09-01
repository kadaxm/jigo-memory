"""
Local voice-cloning TTS server (XTTS-v2, runs in the xtts_env side environment).

- Loads XTTS-v2 once at startup (downloads ~1.8GB on first run)
- GPU (CUDA) if available, CPU fallback; auto-falls back to CPU on CUDA OOM
- Warm-up synthesis at startup so the first real request is fast
- If voice_sample.wav exists in the project root, speaks in the CLONED voice
- Otherwise falls back to a built-in studio speaker (install verification)
- POST /tts {text, language} -> audio/wav (24kHz) + X-Device / X-Synthesis-Ms headers
- GET /health -> {status, model_loaded, cloned, device}

Run (from project root):  xtts_env\\Scripts\\python.exe voice_server.py
"""

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

app = FastAPI(title="Jigo local voice server")
_tts = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_lock = threading.Lock()


def _engine():
    global _tts, _device
    with _lock:
        if _tts is None:
            from TTS.api import TTS
            try:
                _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device)
            except Exception as e:
                print(f"[GPU load failed ({e}) -> falling back to CPU]")
                _device = "cpu"
                _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    return _tts


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
    }


def _synth(text, language, cloned):
    """Synthesize on the active device; on CUDA OOM, move to CPU and retry."""
    engine = _engine()
    with _lock:
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
    """Touch the model once at startup so the first real request isn't slow."""
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
    threading.Thread(target=_warmup, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8100)
