"""
Render the Gujarati line bank with rumik-oss-1 (4-bit, ~0.1x realtime on a 4 GB GPU).

  xtts_env\\Scripts\\python render_gu_bank.py            # render lines that have no wav yet
  xtts_env\\Scripts\\python render_gu_bank.py --redo     # re-render everything
  xtts_env\\Scripts\\python render_gu_bank.py --only sad_01 happy_02

Needs ~2.5 GB free RAM and ~3.4 GB VRAM: close the browser, the voice server and
other GPU apps first. Resumable - each finished line is written immediately.
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from rumik_generate import load_model, synth_one  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--redo", action="store_true")
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--verify", action="store_true",
                help="check each render with ElevenLabs Scribe; retry (up to --tries) until it is heard as Gujarati")
ap.add_argument("--tries", type=int, default=3)
args = ap.parse_args()

FALLBACK_DESC = "warm, gentle, Gujarati accent, slow pace"  # verified to be heard as Gujarati


def _env_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ELEVENLABS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def heard_language(wav_path):
    """(language_code, probability) as detected by ElevenLabs Scribe (free-tier STT)."""
    import httpx
    r = httpx.post("https://api.elevenlabs.io/v1/speech-to-text",
                   headers={"xi-api-key": _env_key()},
                   data={"model_id": "scribe_v1"},
                   files={"file": ("clip.wav", open(wav_path, "rb"), "audio/wav")}, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("language_code"), float(j.get("language_probability") or 0)

AUDIO = ROOT / "gu_bank" / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)
jobs = [json.loads(l) for l in (ROOT / "gu_bank" / "lines.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
if args.only:
    jobs = [j for j in jobs if j["name"] in args.only]
if not args.redo:
    jobs = [j for j in jobs if not (AUDIO / f"{j['name']}.wav").exists()]
print(f"{len(jobs)} lines to render", flush=True)
if not jobs:
    sys.exit(0)

tok, model, mimi = load_model()
t_all = time.time()
for i, j in enumerate(jobs, 1):
    out = AUDIO / f"{j['name']}.wav"
    desc = j["description"]
    for attempt in range(1, (args.tries if args.verify else 1) + 1):
        wav, dur, gen_s = synth_one(tok, model, mimi, j["text"], j.get("speaker", "Ira"), desc)
        sf.write(str(out), wav, 24000)
        msg = f"[{i}/{len(jobs)}] {j['name']}: {dur:.1f}s audio in {gen_s:.0f}s"
        if not args.verify:
            print(msg, flush=True)
            break
        try:
            lang, prob = heard_language(out)
        except Exception as e:  # STT unavailable: keep the render, say so
            print(msg + f"  (verify failed: {str(e)[:60]})", flush=True)
            break
        ok = lang == "guj" and prob >= 0.8
        print(msg + f"  try {attempt}: heard {lang} ({prob:.2f}) {'OK' if ok else 'retry'}  [{desc}]", flush=True)
        if ok:
            break
        desc = FALLBACK_DESC  # later attempts use the style that is known to stay Gujarati
print(f"DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)
