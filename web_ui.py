"""
Jigo web dashboard server.

Serves static/index.html and adds browser-facing endpoints on top of the same
memory core used by the CLI and voice loop. No logic duplicated from memory.py.

Run:  python web_ui.py     ->  http://127.0.0.1:8000
"""

import base64
import os
import re
import tempfile
import time
import wave

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from emotion import analyze_emotion
from llm import _gemini_clients
from jigo_voice import (ANSWER_GATE, REFUSAL, VOICE_ID, looks_like_question,
                        _keyword_intent, compose_reply, eleven_client,
                        transcribe_and_classify_with_fallback)
from memory import TYPE_HALF_LIFE_DAYS, add_memory, collection, search_memory

app = FastAPI(title="Jigo Dashboard")

CLONE_SERVER = "http://127.0.0.1:8100"


def _reply_audio(text):
    """Reply voice: local XTTS clone (your voice, free, private) first;
    ElevenLabs stock voice as fallback when the clone server is down or slow."""
    try:
        import httpx
        r = httpx.post(f"{CLONE_SERVER}/tts", json={"text": text[:600]}, timeout=30.0)
        if r.status_code == 200 and r.content:
            return r.content, "clone", "wav"
    except Exception as e:
        print(f"[clone TTS unavailable ({type(e).__name__}) -> ElevenLabs fallback]")
    audio = eleven_client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_turbo_v2_5",
        optimize_streaming_latency=3,
    )
    return b"".join(audio), "elevenlabs", "mpeg"


class AddRequest(BaseModel):
    content: str
    source: str = "dashboard"


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "memories": collection.count()}


@app.get("/config")
def config():
    """Public values the UI needs (thresholds/half-lives for its decay meters)."""
    return {
        "answer_gate": ANSWER_GATE,
        "half_life_days": TYPE_HALF_LIFE_DAYS,
        "gemini_keys_configured": len(_gemini_clients()),
    }


@app.get("/memories")
def list_memories():
    rows = collection.get(include=["documents", "metadatas"])
    items = []
    for doc_id, doc, meta in zip(rows["ids"], rows["documents"], rows["metadatas"]):
        items.append({
            "id": doc_id,
            "content": doc,
            "salience": round(float(meta.get("salience", 0.5)), 2),
            "type": meta.get("type", "semantic"),
            "timestamp": float(meta.get("timestamp", 0)),
            "source": meta.get("source", "?"),
            "emotion": meta.get("emotion"),
        })
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"memories": items, "count": len(items)}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    collection.delete(ids=[memory_id])
    return {"ok": True}


@app.post("/add")
def add(req: AddRequest):
    memory_id = add_memory(req.content, req.source)
    return {"id": memory_id, "status": "stored"}


def _scribe_transcribe(wav_path):
    """ElevenLabs Scribe STT. Raises on failure (e.g. restricted key)."""
    with open(wav_path, "rb") as f:
        resp = eleven_client.speech_to_text.convert(model_id="scribe_v1", file=f.read())
    return {"intent": _keyword_intent(resp.text or ""), "text": (resp.text or "").strip()}


@app.post("/voice_debug")
async def voice_debug(audio: UploadFile = File(...)):
    """Calibration: what exactly did Scribe hear? (Scribe is production STT;
    Gemini audio remains an invisible fallback, not a lab candidate.)"""
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        out = []
        t0 = time.perf_counter()
        try:
            r = _scribe_transcribe(tmp_path)
            out.append({"engine": "elevenlabs scribe", "ok": True,
                        "transcript": r.get("text", ""), "intent": r.get("intent"),
                        "ms": round((time.perf_counter() - t0) * 1000), "error": None})
        except Exception as e:
            m = str(e)
            why = ("needs full-scope ElevenLabs key"
                   if "missing_permissions" in m else m[:120])
            out.append({"engine": "elevenlabs scribe", "ok": False,
                        "transcript": "", "intent": None,
                        "ms": round((time.perf_counter() - t0) * 1000), "error": why})
        return {"engines": out}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/tts")
def tts(text: str):
    """Short spoken phrases for UI cues (confirmations), as raw mp3."""
    text = text.strip()[:300]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    try:
        mp3 = b"".join(eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            optimize_streaming_latency=3,
        ))
        return Response(content=mp3, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")


@app.get("/tts_stream")
def tts_stream(text: str):
    """Raw PCM (16-bit mono 24kHz) streamed as generated — browser plays chunks live."""
    from fastapi.responses import StreamingResponse
    text = text.strip()[:600]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    def gen():
        yield from eleven_client.text_to_speech.stream(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_24000",
            optimize_streaming_latency=3,
        )

    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.get("/voice_stream")
def voice_stream(text: str):
    """Reply voice, streaming. Clone server (your voice) first — streamed
    sentence-by-sentence; ElevenLabs stock voice as automatic fallback when the
    clone server is down, still loading, or fails before the first byte."""
    from fastapi.responses import StreamingResponse
    text = text.strip()[:600]
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    import httpx

    upstream = None
    try:
        h = httpx.get(f"{CLONE_SERVER}/health", timeout=2.0)
        if h.status_code == 200 and h.json().get("model_loaded"):
            upstream = httpx.stream(
                "POST", f"{CLONE_SERVER}/tts_stream_clone",
                json={"text": text}, timeout=httpx.Timeout(5.0, read=120.0),
            )
            r = upstream.__enter__()
            if r.status_code == 200:
                it = r.iter_bytes()  # single iterator — httpx forbids re-streaming
                first = next(it)     # blocks until first PCM (~2-4s on GPU)

                def gen_clone():
                    try:
                        yield first
                        for chunk in it:
                            yield chunk
                    except Exception as e:
                        print(f"[clone stream aborted mid-way: {e}]")
                    finally:
                        try:
                            upstream.__exit__(None, None, None)
                        except Exception:
                            pass

                return StreamingResponse(gen_clone(), media_type="application/octet-stream")
            upstream.__exit__(None, None, None)
            upstream = None
            print("[clone stream bad status -> ElevenLabs fallback]")
    except Exception as e:
        print(f"[clone stream unavailable ({type(e).__name__}) -> ElevenLabs fallback]")
        if upstream:
            try:
                upstream.__exit__(None, None, None)
            except Exception:
                pass
        upstream = None

    def gen_fallback():
        yield from eleven_client.text_to_speech.stream(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_24000",
            optimize_streaming_latency=3,
        )

    return StreamingResponse(gen_fallback(), media_type="application/octet-stream")


@app.post("/voice")
async def voice(audio: UploadFile = File(...), confirm_store: str = Form("0"), skip_tts: str = Form("0")):
    """Full spoken turn: browser mic audio in -> transcript, routing, memory ops, TTS audio out.

    confirm_store=1 -> store-intents are NOT written; the client shows the transcript
    for human confirmation first (misheard-audio guard). Recalls proceed normally.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)

        t0 = time.perf_counter()
        result = transcribe_and_classify_with_fallback(tmp_path)
        transcribe_ms = (time.perf_counter() - t0) * 1000

        intent = result.get("intent")
        transcript = (result.get("text") or "").strip()
        # strip Scribe's non-speech annotations like [mouse clicking]
        transcript = re.sub(r"\[[^\]]*\]", "", transcript)
        transcript = " ".join(transcript.split())

        # hard guard: pure questions can never be stored, even on intent misfires
        if intent == "store" and looks_like_question(transcript):
            print(f"[intent guard: '{transcript[:50]}' looks like a question -> recall]")
            intent = "recall"

        # speech emotion from the captured audio
        emo = None
        try:
            with wave.open(tmp_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
            pcm = np.frombuffer(frames, dtype=np.int16)
            emo = analyze_emotion(pcm, rate)
        except Exception:
            pass

        if intent == "store" and transcript and confirm_store in ("1", "true", "True"):
            return {
                "intent": "store",
                "transcript": transcript,
                "needs_confirmation": True,
                "stats": {"transcribe_ms": round(transcribe_ms)},
            }

        reply_text = None
        recalled = []
        stored_id = None
        retrieval_ms = None
        gate_score = None
        refused = False
        answer_ms = None

        if intent == "store" and transcript:
            t1 = time.perf_counter()
            stored_id = add_memory(transcript, source="voice",
                                   emotion=(emo or {}).get("label"),
                                   emotion_intensity=(emo or {}).get("intensity"))
            retrieval_ms = round((time.perf_counter() - t1) * 1000)
            reply_text = "Got it, I'll remember that."
        elif intent == "recall" and transcript:
            t1 = time.perf_counter()
            results = search_memory(transcript, top_k=8, associative=False)
            retrieval_ms = round((time.perf_counter() - t1) * 1000)
            gate_score = round(results[0]["raw_similarity"], 3) if results else 0.0
            if results and results[0]["raw_similarity"] >= ANSWER_GATE:
                recalled = [
                    {"content": r["content"], "score": round(r["similarity"], 3),
                     "raw": round(r["raw_similarity"], 3)}
                    for r in results
                ]
                t_answer = time.perf_counter()
                answer = compose_reply(transcript, results)
                answer_ms = round((time.perf_counter() - t_answer) * 1000)
                if answer:
                    reply_text = answer
                    refused = answer.strip() == REFUSAL
                else:
                    # LLM unavailable -> graceful degradation: echo best memory
                    reply_text = results[0]["content"]
            else:
                refused = True
                reply_text = REFUSAL
        else:
            reply_text = "I couldn't tell if that was something to remember or a question."

        audio_bytes = b""
        tts_ms = 0
        voice_source = None
        tts_format = "mpeg"
        if skip_tts not in ("1", "true", "True"):
            t2 = time.perf_counter()
            audio_bytes, voice_source, tts_format = _reply_audio(reply_text)
            tts_ms = (time.perf_counter() - t2) * 1000

        return {
            "intent": intent,
            "transcript": transcript,
            "reply_text": reply_text,
            "refused": refused,
            "gate_score": gate_score,
            "stored_id": stored_id,
            "recalled": recalled,
            "emotion": emo,
            "voice_source": voice_source,
            "stats": {
                "transcribe_ms": round(transcribe_ms),
                "retrieval_ms": retrieval_ms,
                "answer_ms": answer_ms,
                "tts_ms": round(tts_ms),
                "tts_bytes": len(audio_bytes),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            },
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        }
    except HTTPException:
        raise
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            msg = ("Gemini daily free-tier quota exhausted (resets midnight Pacific). "
                   "Transcription is unavailable until then; text add/search still work.")
        raise HTTPException(status_code=503, detail=msg)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    print("Jigo dashboard -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
