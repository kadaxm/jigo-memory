import time

import numpy as np
import sounddevice as sd
import wave
import json
import winsound

from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs

from memory import _gemini_clients, add_memory, search_memory
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

genai_client = genai.Client(api_key=GEMINI_API_KEY)
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


def transcribe_and_classify(filepath):
    with open(filepath, 'rb') as f:
        audio_bytes = f.read()
    prompt = (
        "You are given an audio clip of someone speaking English, Hindi, or a mix. "
        "1) Transcribe exactly what they said, in Roman script if Hindi. "
        "2) Classify intent as 'store' (they are telling you something to remember) "
        "or 'recall' (they are asking a question about something you might already know). "
        'Respond ONLY with JSON in this exact format: {"intent": "store or recall", "text": "cleaned transcription"}'
    )
    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        prompt
    ]
    last_err = None
    for client in (_gemini_clients() or [genai_client]):
        try:
            response = client.models.generate_content(
                model="models/gemini-3-flash-preview",
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
    """Offline intent classification for the quota fallback path."""
    t = text.lower().strip()
    if not t:
        return "recall"
    if t.startswith(("remember", "note that", "keep in mind", "yaad rakho", "yaad rakhna")):
        return "store"
    first = t.split()[0]
    if t.endswith("?") or first in (
        "what", "where", "when", "who", "why", "how", "which",
        "kya", "kahan", "kab", "kaun", "kyun", "kaise",
    ):
        return "recall"
    return "store"  # default: they are telling it something


def transcribe_and_classify_with_fallback(filepath):
    """Gemini first; on daily-quota exhaustion, ElevenLabs Scribe + keyword rules."""
    try:
        return transcribe_and_classify(filepath)
    except Exception as e:
        msg = str(e)
        if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and "quota" not in msg.lower():
            raise
        print("[gemini quota exhausted -> falling back to ElevenLabs Scribe + keyword intent]")
        with open(filepath, 'rb') as f:
            audio_bytes = f.read()
        resp = eleven_client.speech_to_text.convert(
            model_id="scribe_v1",
            file=audio_bytes,
        )
        text = (resp.text or "").strip()
        if text.lower().startswith(("you,", "okay.", "thank you.")) and len(text) < 10:
            text = ""  # scribe sometimes emits filler on near-empty clips
        return {"intent": _keyword_intent(text), "text": text}


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
