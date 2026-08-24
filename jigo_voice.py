import time

import numpy as np
import sounddevice as sd
import wave
import json
import winsound

from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs

import llm
from memory import add_memory, search_memory
from dotenv import load_dotenv
import os

# --- CONFIG (keys live in .env) ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("JIGO_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"  # stock fallback: "George"
# Confidence gate uses RAW embedding similarity, not the hybrid score:
# salience+recency put a ~0.28 floor under every hybrid score, so garbage matches
# still cleared 0.3 easily (eval.py adversarial cases scored 0.61-0.66 hybrid).
# Raw cosine separates cleanly here: legit >= 0.72, off-topic <= 0.64 -> gate at 0.68.
SIMILARITY_THRESHOLD = 0.68
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000  # pcm_24000 output format from the streaming endpoint

genai_client = None  # audio-path fallback clients come from llm._gemini_clients()
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def record_audio(filename="input.wav", seconds=RECORD_SECONDS, rate=SAMPLE_RATE):
    print(f"Recording for {seconds} seconds... speak now.")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    amplitude = np.abs(audio).mean()
    if amplitude < 100:
        print("(silence detected, skipping)")
        return None
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    return filename


def transcribe_and_classify(filepath, model="models/gemini-3.5-flash-lite"):
    with open(filepath, 'rb') as f:
        audio_bytes = f.read()
    prompt = (
        "You are given an audio clip of someone speaking English, Hindi, or a mix. "
        "1) Transcribe exactly what they said, word for word, in Roman script if Hindi. "
        "2) Classify intent as 'store' (they are telling you something to remember) "
        "or 'recall' (they are asking a question about something you might already know). "
        'Respond ONLY with JSON in this exact format: {"intent": "store or recall", "text": "cleaned transcription"}'
    )
    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        prompt
    ]
    last_err = None
    for client in (llm._gemini_clients() or []):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            return json.loads(raw)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                last_err = e
                continue  # this key is out -> try the next
            raise
    if last_err:
        raise last_err
    raise RuntimeError("No Gemini API keys configured")


def _keyword_intent(text):
    """Offline intent classification. Hardened: strips greetings, checks whole string."""
    t = (text or "").lower().strip().rstrip(".,!?").strip()
    if not t:
        return "recall"
    for g in ("hello", "hi", "hey", "ok", "okay", "jigo", "please", "yaar"):
        if t.startswith(g + " ") or t.startswith(g + ","):
            t = t[len(g):].lstrip(" ,.").strip()
    if any(p in t for p in ("remember that", "note that", "keep in mind", "yaad rakho", "yaad rakhna", "make a note")):
        return "store"
    if "?" in t:
        return "recall"
    qwords = ("what", "where", "when", "who", "why", "how", "which",
              "kya", "kahan", "kab", "kaun", "kyun", "kaise")
    if t.split() and t.split()[0] in qwords:
        return "recall"
    if any(w in t for w in qwords):
        return "recall"
    return "store"  # default: they are telling it something


# Recall answer gate on RAW embedding similarity. Relaxed from 0.68 because the
# response layer (compose_reply) now handles ambiguity intelligently — the gate
# only needs to decide "worth trying to answer", not "verbatim match".
ANSWER_GATE = 0.45

REFUSAL = "I don't have anything on that yet."


def compose_reply(query, results):
    """LLM-generated conversational answer grounded ONLY in retrieved memories."""
    mem_lines = "\n".join(f"- {r['content']}" for r in results[:5])
    prompt = (
        "You are Jigo, a warm voice assistant with a memory. The user asks:\n"
        f"\"{query}\"\n\n"
        "Your memories:\n" + mem_lines + "\n\n"
        "Answer conversationally in at most 2 short spoken-style sentences, "
        "using ONLY the memories above. Never invent facts. If the memories do "
        "not contain the answer, respond with exactly: " + REFUSAL
    )
    try:
        out = llm.chat(
            prompt,
            system="Answer directly. No preamble, no reasoning out loud.",
            max_tokens=500,
            temperature=0.5,
        )
        out = " ".join(out.split()).strip()
        return out or None
    except Exception as e:
        print(f"[compose_reply failed ({str(e)[:80]}) -> echoing top memory]")
        return None


def classify_intent_text(text):
    """Intent from text. Strong keyword signals answer instantly (no LLM call);
    ambiguous utterances go to the LLM. Returns None on total failure."""
    t = (text or "").lower().strip()
    if not t:
        return None
    strong_store = any(p in t for p in (
        "remember that", "note that", "keep in mind", "yaad rakho",
        "yaad rakhna", "make a note", "store this",
    ))
    if strong_store:
        return "store"
    if t.endswith("?"):
        return "recall"
    qwords = ("what", "where", "when", "who", "why", "how", "which",
              "kya", "kahan", "kab", "kaun", "kyun", "kaise")
    first = t.split()[0] if t.split() else ""
    if first in qwords:
        return "recall"
    # ambiguous -> LLM
    try:
        r = llm.chat_json(
            'Classify this utterance: "store" (telling you something to remember) or '
            '"recall" (asking a question about what you know). Utterance: "' + text.strip() + '". '
            'Respond ONLY with JSON: {"intent": "store or recall"}'
        )
        intent = r.get("intent")
        return intent if intent in ("store", "recall") else None
    except Exception:
        return None


def transcribe_and_classify_with_fallback(filepath):
    """Primary: ElevenLabs Scribe STT (fast, dedicated) + LLM text intent.
    Fallback: Gemini audio transcription (key-rotating)."""
    try:
        with open(filepath, 'rb') as f:
            audio_bytes = f.read()
        resp = eleven_client.speech_to_text.convert(model_id="scribe_v1", file=audio_bytes)
        text = (resp.text or "").strip()
        if text.lower() in ("you,", "okay.", "thank you.", "bye."):
            text = ""  # scribe filler on near-empty clips
        intent = classify_intent_text(text) or _keyword_intent(text)
        print("[stt: elevenlabs scribe]")
        return {"intent": intent, "text": text}
    except Exception as se:
        print(f"[scribe unavailable ({str(se)[:70]}) -> gemini audio path]")

    try:
        return transcribe_and_classify(filepath)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            print("[gemini quota exhausted on all keys]")
        raise


def speak(text):
    """Stream TTS: playback starts as soon as the first audio chunk arrives."""
    print(f"Jigo: {text}")
    t0 = time.perf_counter()
    try:
        audio_stream = eleven_client.text_to_speech.stream(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format=f"pcm_{TTS_SAMPLE_RATE}",
        )
        player = sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16")
        player.start()
        first_byte_ms = None
        total_bytes = 0
        for chunk in audio_stream:
            if first_byte_ms is None:
                first_byte_ms = (time.perf_counter() - t0) * 1000
            player.write(np.frombuffer(chunk, dtype=np.int16))
            total_bytes += len(chunk)
        player.stop()
        player.close()
        if first_byte_ms is not None:
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[tts: first audio {first_byte_ms:.0f} ms | stream finished {total_ms:.0f} ms | {total_bytes} bytes]")
        else:
            print("[tts: no audio received]")
    except Exception as e:
        # Legacy fallback: generate-then-play whole file.
        print(f"[streaming failed ({e}); falling back to generate-then-play]")
        audio = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
        )
        with wave.open("reply.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            for chunk in audio:
                wf.writeframes(chunk)
        winsound.PlaySound("reply.wav", winsound.SND_FILENAME)


def handle_turn():
    input("\nPress Enter to speak (records 5 seconds)...")
    filepath = record_audio()
    if filepath is None:
        speak("I didn't catch that, try again.")
        return
    result = transcribe_and_classify_with_fallback(filepath)
    intent = result.get("intent")
    text = result.get("text", "").strip()
    print(f"Heard ({intent}): {text}")
    if not text:
        speak("Sorry, I didn't catch that.")
        return
    if intent == "store":
        add_memory(text)
        speak("Got it, I'll remember that.")
    elif intent == "recall":
        t0 = time.perf_counter()
        results = search_memory(text, top_k=1, associative=False)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        top_sim = results[0]["similarity"] if results else 0.0
        print(f"[retrieval: {retrieval_ms:.0f} ms | top match score {top_sim:.2f}]")
        if results and results[0]["raw_similarity"] >= SIMILARITY_THRESHOLD:
            speak(results[0]["content"])
        else:
            speak("I don't have anything on that yet.")
    else:
        speak("Not sure if that was something to remember or a question -- try again.")


if __name__ == "__main__":
    print("Jigo voice loop ready.")
    while True:
        try:
            handle_turn()
        except KeyboardInterrupt:
            print("\nExiting.")
            break
