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
import gu_bank
from emotion_text import query_feeling, text_emotion
from llm import _gemini_clients
from persona import (compose_persona_reply, compose_roast_reply, pick_mode,
                     PERSONA_ACTIVE, ROAST_ACTIVE)
from jigo_voice import (ANSWER_GATE, REFUSAL, VOICE_ID, looks_like_question,
                        _keyword_intent, compose_reply, eleven_client,
                        transcribe_and_classify_with_fallback)
from memory import TYPE_HALF_LIFE_DAYS, add_memory, collection, search_memory

# --- persona content trigger: sadness vocabulary (user says something sad —
# the uncle responds even when the voice tone itself was calm) ---
_SAD_CONTENT = ("maan kharab", "man kharab", "dukh", "udas", "dukhi", "r roy chhu",
                "rou chhu", "low feel", "feeling low", "feeling down", "sad",
                "depressed", "ghabrai", "ghabhray", "dar lag", "anxious", "tension",
                "nusasan", "haani", "nukshan", "loss", "fail", "kharab chhe",
                "kharab chhu", "saras nathi", "barbaad",
                # Scribe mishears Gujarati-accented speech into these English forms
                "not feeling good", "not feeling well", "not feeling",
                # Gujarati native script (Scribe sometimes returns it correctly)
                "મન ખરાબ", "ખરાબ છે", "સરસ નથી", "ઉદાસ", "દુઃખ", "થાકી ગયો",
                "નથી લાગતું")

# happy-content trigger for roast mode (calm speakers: tone says neutral,
# content says celebration)
_HAPPY_CONTENT = ("finally fixed", "finally done", "finally working", "deployed",
                  "shipped", "launched", "got selected", "got placed", "got the job",
                  "got an offer", "offer letter", "cracked the", "cracked my",
                  "cleared the", "passed the", "won the", "got the internship",
                  "got the position", "published", "first place", "got first",
                  "milestone", "it works now", "working now", "solved the")

app = FastAPI(title="Jigo Dashboard")

CLONE_SERVER = "http://127.0.0.1:8100"

_GREET_TOKENS = ("hello", "hi", "hey", "namaste", "namaskar", "kem che", "kem cho",
                 "kem chho", "good morning", "good afternoon", "good evening", "jigo")
_GREET_TAILS = ("how are you", "how r u", "kem cho", "su chale", "kya chal raha",
                "kya haal", "what's up", "whats up")


def _is_greeting(text):
    """Bare greetings and greeting+small-talk ('hello', 'how are you', 'hey bhai
    kem cho') must never be stored as memories."""
    t = " ".join((text or "").lower().split())
    t = re.sub(r"[^\w\s']", "", t).strip()
    if not t:
        return False
    prev = None
    while prev != t:
        prev = t
        if t in _GREET_TOKENS or t in _GREET_TAILS:
            return True
        for g in _GREET_TOKENS:
            if t.startswith(g + " "):
                t = t[len(g) + 1:].strip()
    return t in _GREET_TAILS


def _sad_content(text):
    """Content-level sadness signal: user *says* something sad even in a calm
    voice ('aaj maan kharab chhe bhai...'). Complements emotion.py's tone signal."""
    t = " ".join((text or "").lower().split())
    return any(k in t for k in _SAD_CONTENT)


def _happy_content(text):
    """Content-level celebration signal for roast mode."""
    t = " ".join((text or "").lower().split())
    return any(k in t for k in _HAPPY_CONTENT)


_GU_SCRIPT = re.compile("[઀-૿]")
# romanized Gujarati / Hinglish markers: two or more => speak with Hindi phonology, not English
_ROMAN_INDIC = {"chhe", "chho", "chhu", "chhi", "nathi", "thai", "jase", "bhai", "kem", "aaj", "maan", "kharab",
                "saras", "tame", "tamne", "mane", "badhu", "saru", "bahu", "nahi", "hai", "hain", "kya", "kaise",
                "yaar", "chinta", "kar", "ne", "to", "pan", "ane", "hu", "mari", "tamaru", "su", "shu"}
_INDIC_LANG = os.getenv("JIGO_TTS_INDIC_LANG", "hi")  # "en" disables the Indic path
# "auto": clone first, ElevenLabs fallback. "eleven": skip the clone entirely - on a loaded 4 GB
# laptop the clone ran at 2x slower than realtime (gaps/glitches while streaming).
TTS_ENGINE = os.getenv("JIGO_TTS_ENGINE", "auto")


def _tts_lang_text(text):
    """(text, xtts_language). XTTS-v2 has no Gujarati. Gujarati script is mapped
    codepoint-for-codepoint onto Devanagari (the blocks are parallel) and read with
    Hindi phonology; romanized Gujarati/Hinglish is read as Hindi instead of English
    (which gave an English accent). Plain English stays 'en'."""
    if _INDIC_LANG == "en":
        return text, "en"
    if _GU_SCRIPT.search(text):
        mapped = "".join(chr(ord(c) - 0x180) if "઀" <= c <= "૿" else c for c in text)
        return mapped, _INDIC_LANG
    words = re.findall(r"[a-z']+", text.lower())
    if sum(1 for w in words if w in _ROMAN_INDIC) >= 2:
        return text, _INDIC_LANG
    return text, "en"


_warm = {"t": 0.0}


def _kick_warmup():
    """Load the lazily-unloaded clone model in the background (never blocks a turn)."""
    import threading
    now = time.time()
    if now - _warm["t"] < 150:
        return
    _warm["t"] = now

    def _go():
        try:
            import httpx
            httpx.post(f"{CLONE_SERVER}/warmup", timeout=400.0)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _reply_audio(text):
    """Reply voice: local XTTS clone (your voice, free, private) first;
    ElevenLabs stock voice as fallback when the clone server is down or slow."""
    try:
        if TTS_ENGINE == "eleven":
            raise RuntimeError("clone disabled by JIGO_TTS_ENGINE=eleven")
        import httpx
        ttext, lang = _tts_lang_text(text)
        r = httpx.post(f"{CLONE_SERVER}/tts", json={"text": ttext[:600], "language": lang}, timeout=30.0)
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
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    # no-cache: the browser must always fetch the latest frontend (persona/
    # roast replies, mode fixes) instead of a stale cached copy
    return FileResponse(p, headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                    "Pragma": "no-cache", "Expires": "0"})


@app.get("/health")
def health():
    return {"status": "ok", "memories": collection.count()}


@app.get("/config")
def config():
    """Public values the UI needs (thresholds/half-lives for its decay meters)."""
    try:  # opening the dashboard starts loading the voice clone if its server is up
        import httpx
        h = httpx.get(f"{CLONE_SERVER}/health", timeout=1.0)
        if h.status_code == 200 and not h.json().get("model_loaded"):
            _kick_warmup()
    except Exception:
        pass
    return {
        "answer_gate": ANSWER_GATE,
        "half_life_days": TYPE_HALF_LIFE_DAYS,
        "gemini_keys_configured": len(_gemini_clients()),
        "gujarati_bank": gu_bank.status(),
    }


@app.get("/gu_audio/{line_id}")
def gu_audio(line_id: str):
    """Pre-rendered rumik-oss-1 Gujarati line (see gu_bank.py / render_gu_bank.py)."""
    if not re.fullmatch(r"[a-z]+_[0-9]{2}", line_id):
        raise HTTPException(status_code=404)
    p = gu_bank.audio_path(line_id)
    if not p:
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="audio/wav")


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
    """Confirm-flow store. Persona/roast styling applies here too: a sad
    statement gets the uncle's comfort, a celebration gets the roast."""
    memory_id = add_memory(req.content, req.source)
    reply = "Got it, I'll remember that."
    mode = None
    bank = None
    emo_label, _ = text_emotion(req.content)
    if _sad_content(req.content) and PERSONA_ACTIVE():
        mode = "persona"
        want = emo_label if emo_label in ("sad", "stressed", "angry") else "sad"
        bank = gu_bank.pick(want)
        if bank:
            reply = bank["text"]  # natural Gujarati from rumik-oss, instant (pre-rendered)
            print(f"[gujarati bank: {bank['id']} for {want}]")
        else:
            results = search_memory(req.content, top_k=4, associative=False)
            reply = compose_persona_reply(req.content, results, "sad")
            print("[persona mode: confirm-flow store -> comfort reply (bank empty)]")
    elif emo_label != "neutral" and not (_happy_content(req.content) and ROAST_ACTIVE()):
        bank = gu_bank.pick(emo_label)
        if bank:
            mode, reply = "bank", bank["text"]
            print(f"[gujarati bank: {bank['id']} for {emo_label}]")
    elif _happy_content(req.content) and ROAST_ACTIVE():
        results = search_memory(req.content, top_k=4, associative=False)
        reply = compose_roast_reply(req.content, results, "happy")
        mode = "roast"
        print("[roast mode: confirm-flow store -> demotivator reply]")
    return {"id": memory_id, "status": "stored",
            "reply_text": reply if mode else None, "reply_mode": mode, "bank": bank,
            "emotion": emo_label if emo_label != "neutral" else None}


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
        if TTS_ENGINE == "eleven":
            raise RuntimeError("clone disabled by JIGO_TTS_ENGINE=eleven")
        h = httpx.get(f"{CLONE_SERVER}/health", timeout=2.0)
        if h.status_code == 200 and not h.json().get("model_loaded"):
            _kick_warmup()  # this turn falls back to ElevenLabs; the next one uses the clone
            print("[clone model not loaded -> warming up in background, ElevenLabs for this turn]")
        if h.status_code == 200 and h.json().get("model_loaded"):
            ttext, lang = _tts_lang_text(text)
            upstream = httpx.stream(
                "POST", f"{CLONE_SERVER}/tts_stream_clone",
                json={"text": ttext, "language": lang}, timeout=httpx.Timeout(5.0, read=120.0),
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
async def voice(audio: UploadFile = File(...), confirm_store: str = Form("0"), skip_tts: str = Form("0"),
                lang: str = Form("auto")):
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
        # capture input duration — separates "long recording" from "slow API"
        # when transcribe_ms spikes (28s mystery diagnosis)
        try:
            with wave.open(tmp_path, "rb") as wf:
                audio_s = wf.getnframes() / float(wf.getframerate() or 16000)
        except Exception:
            audio_s = -1
        print(f"[stt input: {audio_s:.1f}s audio]")
        result = transcribe_and_classify_with_fallback(
            tmp_path, language={"gu": "guj", "hi": "hin", "en": "eng"}.get(lang))
        transcribe_ms = (time.perf_counter() - t0) * 1000
        print(f"[stt: {transcribe_ms}ms for {audio_s:.1f}s audio]")

        intent = result.get("intent")
        transcript = (result.get("text") or "").strip()
        # strip Scribe's non-speech annotations like [mouse clicking]
        transcript = re.sub(r"\[[^\]]*\]", "", transcript)
        transcript = " ".join(transcript.split())

        # hard guard: pure questions can never be stored, even on intent misfires
        if intent == "store" and looks_like_question(transcript):
            print(f"[intent guard: '{transcript[:50]}' looks like a question -> recall]")
            intent = "recall"

        # greeting/small-talk guard: greetings get a friendly reply, never stored
        # (the keyword classifier defaults to store, which swallowed greetings)
        if _is_greeting(transcript):
            print(f"[intent guard: greeting '{transcript[:40]}' -> friendly reply, not stored]")
            intent = "greeting"

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
        emo_label = (emo or {}).get("label")

        # persona / roast mode: user-mood triggers a styled reply instead of
        # the neutral grounded answer (uncle calms sadness, roast teases joy)
        # trigger = detected TONE (emotion.py) OR sad/celebration CONTENT (user
        # *says* something sad/good in a calm voice)
        content_sad = _sad_content(transcript)
        content_happy = _happy_content(transcript)
        reply_mode = pick_mode(emo_label)
        if not reply_mode and content_sad and PERSONA_ACTIVE():
            reply_mode = "persona"
            print(f"[persona mode: sad CONTENT detected in '{transcript[:50]}']")
        if not reply_mode and content_happy and ROAST_ACTIVE():
            reply_mode = "roast"
            print(f"[roast mode: happy CONTENT detected in '{transcript[:50]}']")

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
            if reply_mode == "persona":
                # uncle acknowledged the update AND comforted: store + reply persona
                results = search_memory(transcript, top_k=4, associative=False)
                reply_text = compose_persona_reply(transcript, results, emo_label)
                print(f"[persona mode: sad statement STORED + uncle comfort reply]")
            elif reply_mode == "roast":
                results = search_memory(transcript, top_k=4, associative=False)
                reply_text = compose_roast_reply(transcript, results, emo_label)
                print(f"[roast mode: happy statement STORED + roast reply]")
            else:
                reply_text = "Got it, I'll remember that."
        elif intent == "unclear_script":
            # Scribe mis-detected the language (spoken Gujarati -> Cyrillic);
            # Gemini retry garbled it too. Honest clarification, nothing stored.
            reply_text = ("I couldn't catch that clearly — Gujarati speech isn't "
                          "supported by my transcription yet. Try again in English "
                          "or Hinglish, and I'll answer in Gujarati.")
        elif intent == "greeting":
            reply_text = "Hey! Jigo here — tell me something to remember, or ask me anything."
        elif intent == "recall" and transcript:
            t1 = time.perf_counter()
            feeling = query_feeling(transcript)  # "what has been stressing me out?" -> "stressed"
            results = search_memory(transcript, top_k=8, associative=False, feeling=feeling)
            retrieval_ms = round((time.perf_counter() - t1) * 1000)
            gate_score = round(results[0]["raw_similarity"], 3) if results else 0.0
            # feeling questions share no words with their answers: the emotion tag is the evidence
            if results and (results[0]["raw_similarity"] >= ANSWER_GATE or results[0].get("emotion_match")):
                recalled = [
                    {"content": r["content"], "score": round(r["similarity"], 3),
                     "raw": round(r["raw_similarity"], 3)}
                    for r in results
                ]
                t_answer = time.perf_counter()
                if reply_mode == "persona":
                    reply_text = compose_persona_reply(transcript, results, emo_label)
                elif reply_mode == "roast":
                    reply_text = compose_roast_reply(transcript, results, emo_label)
                else:
                    reply_text = compose_reply(transcript, results)
                answer_ms = round((time.perf_counter() - t_answer) * 1000)
                if not reply_text:
                    # LLM unavailable -> graceful degradation: echo best memory
                    reply_text = results[0]["content"]
                refused = reply_text.strip() == REFUSAL
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
            "reply_mode": reply_mode,
            "voice_source": voice_source,
            "stats": {
                "transcribe_ms": round(transcribe_ms),
                "audio_s": round(audio_s, 1),
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
