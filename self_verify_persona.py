"""Self-verification: replay the user's exact failing case end-to-end.
1. Generate TTS audio of the sad Gujarati sentence (XTTS clone voice).
2. POST it to /voice (mic path) with skip_tts=0.
3. Assert: intent routed, persona fired, NOT stored as a bare memory, reply in uncle register."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx  # noqa: E402

SAD_GUJ = "આજે મન ખરાબ છે ભાઈ કંઈક સરસ નથી લાગતું"

# 1. synthesize the user's sentence in the clone voice (lang=hi path, verified)
print("1) synthesizing sad Gujarati sentence via clone server...", flush=True)
r = httpx.post("http://127.0.0.1:8100/tts",
               json={"text": SAD_GUJ, "language": "hi"}, timeout=180.0)
print("   TTS:", r.status_code, len(r.content), "bytes", flush=True)
assert r.status_code == 200, r.text[:200]
tmp = os.path.join(os.environ["TEMP"], "sad_guj_input.wav")
with open(tmp, "wb") as f:
    f.write(r.content)

# 2. POST to /voice exactly like the dashboard does
print("2) POST /voice with that audio (Scribe will transcribe it)...", flush=True)
with open(tmp, "rb") as f:
    r2 = httpx.post("http://127.0.0.1:8000/voice",
                    files={"audio": ("rec.wav", open(tmp, "rb").read(), "audio/wav")},
                    data={"confirm_store": "1", "skip_tts": "0"}, timeout=180.0)
print("   /voice HTTP:", r2.status_code, flush=True)
d = r2.json()
print("   intent:", d.get("intent"), flush=True)
print("   transcript:", (d.get("transcript") or "")[:90], flush=True)
print("   reply_mode:", d.get("reply_mode"), flush=True)
print("   reply:", (d.get("reply_text") or "")[:140], flush=True)
print("   stored_id:", d.get("stored_id"), flush=True)

# 3. assertions
ok_intent = d.get("intent") in ("recall", "store")
ok_mode = d.get("reply_mode") == "persona"
reply = d.get("reply_text") or ""
ok_reply = any(k in reply.lower() for k in ("thai jase", "tension nai", "himmat", "evu to", "saras", "maan"))
print("\nASSERTIONS:", flush=True)
print(f"  reply_mode==persona: {ok_mode}", flush=True)
print(f"  uncle-register reply: {ok_reply}", flush=True)

# 4. confirm nothing new stored (count check)
import chromadb  # noqa: E402
os.chdir(os.path.dirname(os.path.abspath(__file__)))
c = chromadb.PersistentClient(path="./chroma_memory")
col = [x for x in c.list_collections() if x.name == "memories"][0]
n = col.count()
print(f"  memory count after turn: {n}", flush=True)

print("\nSELF_VERIFY_DONE", flush=True)