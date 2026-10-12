"""Self-verify: replay the user's exact demo sentences through the live stack.
For each: synth via clone server (their voice) -> POST /voice -> assert mode fired."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx  # noqa: E402

CASES = [
    {"name": "sad_english", "input": "I am not feeling good today", "input_lang": "en",
     "tts_desc": "sad, slow pace", "expect": "persona", "need": ("thai jase", "tension", "himmat", "evu to", "saras")},
    {"name": "happy_roast", "input": "Bro, I finally fixed the bug and deployed today", "input_lang": "en",
     "tts_desc": "excited, fast pace", "expect": "roast", "need": ("arey", "server", "crash", "countdown", "thik")},
]

tmp = os.path.join(os.environ["TEMP"], "selfverify")
os.makedirs(tmp, exist_ok=True)
for c in CASES:
    print(f"--- {c['name']}: synthesizing input via clone ({c['tts_desc']}) ---", flush=True)
    r = httpx.post("http://127.0.0.1:8100/tts",
                   json={"text": c["input"], "language": c["input_lang"]}, timeout=180.0)
    print("  TTS:", r.status_code, len(r.content), flush=True)
    assert r.status_code == 200
    p = os.path.join(tmp, c["name"] + "_in.wav")
    with open(p, "wb") as f:
        f.write(r.content)

    print(f"  POST /voice...", flush=True)
    with open(p, "rb") as f:
        r2 = httpx.post("http://127.0.0.1:8000/voice",
                        files={"audio": ("rec.wav", f.read(), "audio/wav")},
                        data={"confirm_store": "1", "skip_tts": "1"}, timeout=180.0)
    d = r2.json()
    transcript = (d.get("transcript") or "")
    mode = d.get("reply_mode")
    reply = d.get("reply_text") or ""
    print(f"  transcript: {transcript[:80]}", flush=True)
    print(f"  reply_mode: {mode}", flush=True)
    print(f"  reply: {reply[:120]}", flush=True)

    if c["expect"] == "persona":
        hit = mode == "persona" and any(k in reply.lower() for k in c["need"])
    else:
        hit = mode == "roast" and len(reply) > 5
    print(f"  {'✅ PASS' if hit else '❌ FAIL'}", flush=True)

print("SELF_VERIFY_DONE", flush=True)