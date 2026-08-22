"""
Jigo web dashboard server.

Serves static/index.html and adds browser-facing endpoints on top of the same
memory core used by the CLI and voice loop. No logic duplicated from memory.py.

Run:  python web_ui.py     ->  http://127.0.0.1:8000
"""

import base64
import os
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from jigo_voice import (SIMILARITY_THRESHOLD, VOICE_ID, _gemini_clients,
                        eleven_client, transcribe_and_classify_with_fallback)
from memory import TYPE_HALF_LIFE_DAYS, add_memory, collection, search_memory

app = FastAPI(title="Jigo Dashboard")


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
        "similarity_threshold": SIMILARITY_THRESHOLD,
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
            model_id="eleven_multilingual_v2",
        ))
        return Response(content=mp3, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")


@app.post("/voice")
async def voice(audio: UploadFile = File(...), confirm_store: str = Form("0")):
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
        memory_ms = None
        gate_score = None
        refused = False

        t1 = time.perf_counter()
        if intent == "store" and transcript:
            stored_id = add_memory(transcript, source="voice")
            reply_text = "Got it, I'll remember that."
        elif intent == "recall" and transcript:
            results = search_memory(transcript, top_k=3, associative=False)
            if results:
                gate_score = round(results[0]["raw_similarity"], 3)
                if results[0]["raw_similarity"] >= SIMILARITY_THRESHOLD:
                    reply_text = results[0]["content"]
                    recalled = [
                        {"content": r["content"], "score": round(r["similarity"], 3),
                         "raw": round(r["raw_similarity"], 3)}
                        for r in results
                    ]
                else:
                    refused = True
                    reply_text = "I don't have anything on that yet."
            else:
                refused = True
                reply_text = "I don't have anything yet."
        else:
            reply_text = "I couldn't tell if that was something to remember or a question."
        memory_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        audio_gen = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=reply_text,
            model_id="eleven_multilingual_v2",
        )
        mp3 = b"".join(audio_gen)
        tts_ms = (time.perf_counter() - t2) * 1000

        return {
            "intent": intent,
            "transcript": transcript,
            "reply_text": reply_text,
            "refused": refused,
            "gate_score": gate_score,
            "stored_id": stored_id,
            "recalled": recalled,
            "stats": {
                "transcribe_ms": round(transcribe_ms),
                "memory_ms": round(memory_ms),
                "tts_ms": round(tts_ms),
                "tts_bytes": len(mp3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            },
            "audio_b64": base64.b64encode(mp3).decode("ascii"),
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
