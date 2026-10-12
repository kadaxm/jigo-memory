"""
Objective check of the rendered Gujarati bank: ask ElevenLabs Scribe which language
it hears in each clip. A line is OK when Scribe says Gujarati (guj) with >= 0.8
confidence; anything else (hin/mar/nep...) means the model spoke it with another
language's phonology and the line should be re-rendered with a different description.

  python verify_gu_bank.py            # all rendered lines
  python verify_gu_bank.py happy_01   # specific lines
Uses ElevenLabs free-tier STT credits (~3 s of audio per line).
"""
import json
import os
import pathlib
import sys

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

ROOT = pathlib.Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(ROOT / ".env")
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
want = set(sys.argv[1:])
bad = []
VERIFIED = ROOT / "gu_bank" / "verified.json"
verified = json.loads(VERIFIED.read_text(encoding="utf-8")) if VERIFIED.exists() else {}
for l in (ROOT / "gu_bank" / "lines.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    d = json.loads(l)
    wav = ROOT / "gu_bank" / "audio" / f"{d['name']}.wav"
    if not wav.exists() or (want and d["name"] not in want):
        continue
    r = client.speech_to_text.convert(model_id="scribe_v1", file=wav.read_bytes())
    prob = getattr(r, "language_probability", 0) or 0
    ok = r.language_code == "guj" and prob >= 0.8
    print(f"{'OK ' if ok else 'BAD'} {d['name']:12s} heard={r.language_code} ({prob:.2f})  {d['description']}")
    verified[d["name"]] = {"heard": r.language_code, "prob": round(prob, 2), "ok": ok}
    if not ok:
        bad.append(d["name"])
VERIFIED.write_text(json.dumps(verified, indent=1), encoding="utf-8")
print(f"\nwrote {VERIFIED.name}: the dashboard only plays lines marked ok")
print("Re-render these:" if bad else "All lines detected as Gujarati.", " ".join(bad))
