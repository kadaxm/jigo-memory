"""
One-time voice cloning setup.

Records ~60 seconds of your speech via sounddevice, sends it to ElevenLabs
Instant Voice Cloning, prints the resulting voice_id, and saves it to .env as
JIGO_VOICE_ID so jigo_voice.py picks it up automatically.

Run from the project folder:  python clone_voice.py
Speak naturally the whole time — reading a book/article aloud works well.
"""

import os
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

SAMPLE_PATH = "voice_sample.wav"
SECONDS = 60
SAMPLE_RATE = 16000


def record_sample():
    print(f"\nRecording {SECONDS} seconds — START SPEAKING NOW.")
    frames = []
    chunk_seconds = 5
    for i in range(SECONDS // chunk_seconds):
        audio = sd.rec(int(chunk_seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="int16")
        sd.wait()
        frames.append(audio)
        level = np.abs(audio).mean()
        bar = "#" * int(min(level, 3000) / 100)
        print(f"  {5 * (i + 1):3d}s  level {level:6.0f} {bar}")
    if np.abs(np.concatenate(frames)).mean() < 100:
        raise RuntimeError("Recording was near-silent — check your mic and retry.")

    with wave.open(SAMPLE_PATH, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.concatenate(frames).tobytes())
    print(f"Saved {SAMPLE_PATH} ({os.path.getsize(SAMPLE_PATH)} bytes).")


def clone_voice():
    load_dotenv()
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    with open(SAMPLE_PATH, "rb") as f:
        response = client.voices.ivc.create(
            name="Jigo Owner",
            files=[f],
            remove_background_noise=True,
        )
    voice_id = response.voice_id
    print(f"\nCloned! voice_id = {voice_id}")

    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = [ln for ln in f.read().splitlines() if not ln.startswith("JIGO_VOICE_ID=")]
    lines.append(f"JIGO_VOICE_ID={voice_id}")
    with open(env_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved JIGO_VOICE_ID to {env_path} — jigo_voice.py will use it on next run.")


if __name__ == "__main__":
    try:
        input("Press Enter to start recording (speak for the full 60 seconds)...")
    except EOFError:
        import time
        for remaining in range(10, 0, -1):
            print(f"Starting in {remaining}s... get ready to SPEAK for 60 seconds.")
            time.sleep(1)
    record_sample()
    clone_voice()
