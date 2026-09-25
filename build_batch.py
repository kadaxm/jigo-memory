import json
import os

os.chdir(r"C:\Users\kadam\Documents\Jigo-memory")
out = []
with open("benchmarks/gujarati_indicemo_lite/prompts.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        out.append({"name": p["id"] + "_delivery", "text": p["text"], "speaker": "Ira",
                    "description": f"{p['delivery']}, Gujarati accent, {p['pace']} pace"})
        out.append({"name": p["id"] + "_neutral", "text": p["text"], "speaker": "Ira",
                    "description": "neutral, steady pace"})
with open("benchmarks/gujarati_indicemo_lite/batch.jsonl", "w", encoding="utf-8") as f:
    for j in out:
        f.write(json.dumps(j, ensure_ascii=False) + "\n")
print(f"batch file written: {len(out)} jobs")
