"""Export (text, salience) pairs from ChromaDB for salience-model training."""
import json
import os
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import chromadb

client = chromadb.PersistentClient(path=os.path.join(PROJECT, "chroma_memory"))
col = client.get_or_create_collection("memories", metadata={"hnsw:space": "cosine"})

rows = col.get(include=["documents", "metadatas"])
judged, heuristic = [], []
seen = set()
for doc, meta in zip(rows["documents"], rows["metadatas"]):
    meta = meta or {}
    s = meta.get("salience")
    src = str(meta.get("source", ""))
    if doc in seen or s is None:
        continue
    seen.add(doc)
    item = {"text": doc, "salience": float(s), "source": src, "type": meta.get("type", "")}
    # heuristic defaults: obsidian import (0.6) and seed scripts (0.5) are NOT LLM judgments
    if src == "obsidian" or src.startswith("obsidian:"):
        heuristic.append(item)
    else:
        judged.append(item)

with open(os.path.join(PROJECT, "salience_dataset.json"), "w", encoding="utf-8") as f:
    json.dump({"judged": judged, "heuristic": heuristic}, f, ensure_ascii=False)

print(f"LLM-judged examples: {len(judged)}")
print(f"heuristic-only (obsidian/seeds): {len(heuristic)}")
from collections import Counter
print("judged salience distribution:", sorted(Counter(round(x['salience'],1) for x in judged).items()))
print("judged sources:", Counter(x['source'] for x in judged).most_common(8))
