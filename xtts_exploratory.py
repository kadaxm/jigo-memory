"""XTTS exploratory arm v2 (fixed transformers stack).
Arms: (a) Gujarati native script via lang=hi (surprising: works), (b) romanized
via lang=hi, digits pre-expanded (num2words lacks 'hi')."""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx  # noqa: E402

BASE = "http://127.0.0.1:8100"
OUT = r"C:\Users\kadam\Documents\Jigo-memory\benchmarks\gujarati_indicemo_lite\xtts_audio"
os.makedirs(OUT, exist_ok=True)

JOBS = [
    {"name": "xtts_script_gu0001", "text": "મમ્મી, મને placement મળી ગઈ! તમે કહ્યું હતું ને hard work કદી waste નઈ જાય.", "language": "hi"},
    {"name": "xtts_script_gu0006", "text": "દાદા નું ઘર હવે ખાલી લાગે છે. એમના chai ની વાસણો હજી kitchen માં રાખી છે.", "language": "hi"},
    {"name": "xtts_script_gu0011", "text": "મેં ત્રણ વખત કહ્યું — project deadline આવે છે! છતાં team એ કંઈ પૂછ્યું જ નહીં.", "language": "hi"},
    {"name": "xtts_script_gu0016", "text": "મને hackathon માં પહેલું સ્થાન! એટલા કલાક no-sleep, છતાં મારો demo judges ને સૌથી ગમ્યો!", "language": "hi"},
    {"name": "xtts_roman_gu0001", "text": "Mummy, mane placement mali gai! Tame kahyu hatu ne hard work kadi waste nai jay.", "language": "hi"},
    {"name": "xtts_roman_gu0006", "text": "Dada nu ghar have khali lage chhe. Emna chai ni vasano haji pana kitchen ma rakhi chhe.", "language": "hi"},
    {"name": "xtts_roman_gu0011", "text": "Me tran varath kahyu — project deadline aavati chhe! Chhata team e kau puchhyu j nahi.", "language": "hi"},
    {"name": "xtts_roman_gu0016", "text": "Mane hackathon ma pahelu sthan, adatalis kalak no sleep, chhata maro demo judges ne sauthi gamyo.", "language": "hi"},
]

results = []
for job in JOBS:
    t0 = time.time()
    entry = {"name": job["name"], "language": job["language"], "status": None}
    try:
        r = httpx.post(f"{BASE}/tts", json={"text": job["text"], "language": job["language"]}, timeout=180.0)
        entry["http"] = r.status_code
        entry["ms"] = round((time.time() - t0) * 1000)
        if r.status_code == 200 and r.content:
            path = os.path.join(OUT, job["name"] + ".wav")
            with open(path, "wb") as f:
                f.write(r.content)
            entry["status"] = "generated"
            entry["bytes"] = len(r.content)
            entry["path"] = path
        else:
            entry["status"] = f"http_{r.status_code}"
            entry["detail"] = r.text[:200]
    except Exception as e:
        entry["status"] = f"exception: {type(e).__name__}"
        entry["detail"] = str(e)[:200]
    results.append(entry)
    print(f"{job['name']}: {entry['status']} {entry.get('ms','')}ms {entry.get('bytes','')}", flush=True)

with open(os.path.join(OUT, "xtts_attempts.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("XTTS_EXPLORATORY_DONE", flush=True)