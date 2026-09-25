"""Judge panel (substituted, preliminary): free Gemini audio judges rate
delivery-conditioned Gujarati clips 1-5 vs neutral baseline (same text).
Uses llm._gemini_clients() keys; median across available judges."""
import base64
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT = r"C:\Users\kadam\Documents\Jigo-memory"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from llm import _gemini_clients, GEMINI_TEXT_MODEL, GEMINI_FAST_MODEL  # noqa: E402
from google.genai import types  # noqa: E402

BD = os.path.join(PROJECT, "benchmarks", "gujarati_indicemo_lite")
AUDIO = os.path.join(BD, "audio")
OUTD = os.path.join(BD, "judgments")
os.makedirs(OUTD, exist_ok=True)

prompts = {}
with open(os.path.join(BD, "prompts.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            p = json.loads(line)
            prompts[p["id"]] = p

clients = _gemini_clients()
print(f"gemini clients available: {len(clients)}", flush=True)
if not clients:
    print("NO JUDGES CONFIGURED", flush=True)
    sys.exit(1)

JUDGE_MODEL = GEMINI_FAST_MODEL  # higher free-tier quota; audio-capable
print(f"judge model: {JUDGE_MODEL}", flush=True)


def judge_clip(client, delivery, wav_path):
    with open(wav_path, "rb") as f:
        b = f.read()
    prompt = (
        f"You are judging expressive delivery in Gujarati speech (code-switched with English). "
        f"Requested delivery: {delivery.upper()}. Rate ONLY the expressive delivery quality "
        f"on this anchored 1-5 scale:\n"
        f"1 = flat/monotone, no requested delivery\n"
        f"2 = barely perceptible delivery\n"
        f"3 = recognizable but weak delivery\n"
        f"4 = clear, convincing delivery\n"
        f"5 = strong, sustained, natural performance\n"
        f"Axes: tone match, dynamics, phrasing, sustained expression. "
        f'Respond ONLY with JSON: {{"score": <1-5 float>, "note": "<max 10 words>"}}'
    )
    contents = [
        types.Part.from_bytes(data=b, mime_type="audio/wav"),
        prompt,
    ]
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=100),
    )
    raw = (resp.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    return json.loads(raw)


results = []
quota_hit = False
for pid, p in sorted(prompts.items()):
    wav = os.path.join(AUDIO, f"{pid}_delivery.wav")
    if not os.path.exists(wav):
        print(f"{pid}: no delivery wav, skip", flush=True)
        continue
    for ci, client in enumerate(clients):
        rec = {"id": pid, "delivery": p["delivery"], "judge": f"gemini_{ci}"}
        try:
            t0 = time.time()
            out = judge_clip(client, p["delivery"], wav)
            rec.update({"score": out.get("score"), "note": out.get("note", ""), "ok": True})
        except Exception as e:
            msg = str(e)
            rec.update({"ok": False, "error": msg[:150]})
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                quota_hit = True
        rec["ms"] = round((time.time() - t0) * 1000)
        results.append(rec)
        print(f"{pid} [{p['delivery']}] judge{ci}: score={rec.get('score')} ({rec['ms']}ms)", flush=True)
        time.sleep(1)

with open(os.path.join(OUTD, "judgments_raw.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

ok = [r for r in results if r.get("ok") and r.get("score") is not None]
print(f"\njudged: {len(ok)}/{len(results)} calls ok | quota_hit={quota_hit}", flush=True)
if ok:
    import statistics
    scores = [r["score"] for r in ok]
    print(f"overall mean score: {statistics.mean(scores):.2f} (n={len(scores)})", flush=True)
print("JUDGE_RUN_DONE", flush=True)