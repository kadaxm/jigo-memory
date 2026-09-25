"""Live test: persona modes (uncle + roast) + classic compose_reply unchanged."""
import os
import sys

PROJECT = r"C:\Users\kadam\Documents\Jigo-memory"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from persona import compose_persona_reply, compose_roast_reply, pick_mode, PERSONA_MODE, ROAST_MODE  # noqa: E402
from jigo_voice import compose_reply  # noqa: E402

print(f"PERSONA_MODE={PERSONA_MODE} ROAST_MODE={ROAST_MODE}", flush=True)

# mode routing
print(f"pick_mode(sad) = {pick_mode('sad')}      (expect persona)", flush=True)
print(f"pick_mode(stressed) = {pick_mode('stressed')} (expect persona)", flush=True)
print(f"pick_mode(happy) = {pick_mode('happy')}    (expect None unless roast on)", flush=True)
print(f"pick_mode(neutral) = {pick_mode('neutral')}  (expect None)", flush=True)

MEM = [
    {"content": "Movie night moved to Wednesday at 2 PM", "timestamp": 1756672800.0},
    {"content": "I am building Jigo, a voice-native memory assistant", "timestamp": 1756400000.0},
]

print("\n--- uncle reply (sad user) ---", flush=True)
try:
    r = compose_persona_reply("I am feeling low today, the project demo did not go well", MEM, "sad")
    print("persona:", r, flush=True)
except Exception as e:
    print("persona FAILED:", e, flush=True)

print("\n--- roast reply (happy user, toggle ON via call) ---", flush=True)
try:
    r = compose_roast_reply("Bro I finally fixed the bug and deployed everything today!", MEM, "happy")
    print("roast:", r, flush=True)
except Exception as e:
    print("roast FAILED:", e, flush=True)

print("\n--- classic compose_reply (untouched path) ---", flush=True)
try:
    r = compose_reply("when is my movie night?", MEM)
    print("classic:", r, flush=True)
except Exception as e:
    print("classic FAILED:", e, flush=True)

print("\nDONE", flush=True)
