"""Distill LLM salience judgments onto a diverse sample of vault chunks.

Creates the training set for the salience model: the same judge prompt used in
add_memory, run over a random sample of imported notes, at temperature 0.
"""
import json
import os
import random
import sys
import time

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT, ".env"))

import llm

SAMPLE_N = 400
DATA_PATH = os.path.join(PROJECT, "salience_dataset.json")

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

judged = data["judged"]
already = {x["text"] for x in judged}
pool = [x for x in data["heuristic"] if len(x["text"].split()) >= 5 and x["text"] not in already]
random.seed(42)
sample = random.sample(pool, min(SAMPLE_N, len(pool)))
print(f"Judging {len(sample)} chunks (pool {len(pool)}) + {len(judged)} existing judged...")

ok, failed = 0, 0
t0 = time.time()
for i, item in enumerate(sample):
    prompt = (
        f'Rate how important this memory is to remember long-term, from 0.0 (trivial, forgettable) '
        f'to 1.0 (critical, identity-defining). Content: "{item["text"][:600]}". '
        'Respond ONLY with JSON: {"salience": 0.0}'
    )
    try:
        r = llm.chat_json(prompt, max_tokens=200, temperature=0.0, timeout=20, model_hint="fast_gemini")
        s = float(r.get("salience", -1))
        if not (0.0 <= s <= 1.0):
            raise ValueError("out of range")
        item["salience"] = s
        judged.append(item)
        ok += 1
    except Exception:
        failed += 1
    if (i + 1) % 25 == 0:
        data["judged"] = judged
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  {i+1}/{len(sample)} judged ({ok} ok, {failed} failed) — {time.time()-t0:.0f}s [checkpoint saved]")

data["judged"] = judged
with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"DONE: +{ok} new judgments, {failed} failed. Total judged dataset: {len(judged)}")
