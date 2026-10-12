"""Gujarati intelligibility of rumik-oss clips: character error rate (CER) of an
ASR transcript against the prompt text, for delivery and neutral clips.

ASR: free-tier Gemini (audio in -> verbatim transcript, English words kept in
Latin script so they match the prompt text). CER is on lowercased,
punctuation-stripped text. An earlier run that let Gemini transliterate English
words into Gujarati script gave CER > 1 (a metric artifact, not intelligibility).
Resumable; results cached in judgments/asr_transcripts_v2.json.

Run: python benchmarks/gujarati_indicemo_lite/intelligibility_cer.py
"""
import json
import os
import pathlib
import re
import statistics as st
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BD = pathlib.Path(__file__).resolve().parent
PROJECT = BD.parents[1]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

from llm import _gemini_clients, GEMINI_FAST_MODEL  # noqa: E402
from google.genai import types  # noqa: E402

CACHE = BD / "judgments" / "asr_transcripts_v2.json"
cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
prompts = {}
for line in open(BD / "prompts.jsonl", encoding="utf-8"):
    if line.strip():
        p = json.loads(line)
        prompts[p["id"]] = p

clients = _gemini_clients()
GU = re.compile(r"[઀-૿]")


def norm(s):
    """lowercase, drop punctuation/dashes, collapse whitespace; keep Gujarati + Latin letters."""
    s = re.sub(r"[^\w\s઀-૿]", " ", (s or "").lower())
    return " ".join(s.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def transcribe(wav):
    b = open(wav, "rb").read()
    prompt = ("Transcribe this code-switched Gujarati/English speech verbatim. "
              "Write Gujarati words in Gujarati script and English words in Latin "
              "script exactly as spoken (do not transliterate English words into "
              "Gujarati script, do not translate). Output only the transcript.")
    last = None
    for c in clients:
        try:
            r = c.models.generate_content(
                model=GEMINI_FAST_MODEL,
                contents=[types.Part.from_bytes(data=b, mime_type="audio/wav"), prompt],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=300),
            )
            return (r.text or "").strip()
        except Exception as e:
            last = e
    raise last


for pid in sorted(prompts):
    for kind in ("delivery", "neutral"):
        key = f"{pid}_{kind}"
        if cache.get(key):
            continue
        try:
            cache[key] = transcribe(BD / "audio" / f"{key}.wav")
            print(key, "ok", flush=True)
        except Exception as e:
            print(key, "FAILED", str(e)[:70], flush=True)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        time.sleep(4)

res = {"delivery": [], "neutral": []}
by_cat = {}
for pid, p in prompts.items():
    ref = norm(p["text"])
    for kind in ("delivery", "neutral"):
        hyp = cache.get(f"{pid}_{kind}")
        if hyp is None or not ref:
            continue
        cer = edit_distance(ref, norm(hyp)) / len(ref)
        res[kind].append(cer)
        by_cat.setdefault((p["delivery"], kind), []).append(cer)

print("\nCER vs prompt text, ASR = free Gemini (lower is better)")
for kind in ("delivery", "neutral"):
    if res[kind]:
        print(f"{kind:9s} mean={st.mean(res[kind]):.3f} median={st.median(res[kind]):.3f} n={len(res[kind])}")
for (c, k), v in sorted(by_cat.items()):
    print(f"  {c:13s}{k:9s} mean={st.mean(v):.3f} n={len(v)}")
