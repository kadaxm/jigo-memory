"""
Type-to-test Jigo: the real store and recall pipeline without a microphone.

  python try_jigo.py            # SAFE: throwaway in-memory database (your real memories untouched)
  python try_jigo.py --real     # your real chroma_memory (new test rows persist; delete them in the ledger)

Type a statement to store it, or a question to recall. Prefix with "add:" to force a store.
Every turn prints what the pipeline did: emotion tag, gate score, retrieved memories, answer.
Commands:  /list   /reset   /quit
"""
import sys
import time

import chromadb

import memory as M
from emotion_text import query_feeling, text_emotion
from jigo_voice import ANSWER_GATE, REFUSAL, classify_intent_text, compose_reply, looks_like_question

REAL = "--real" in sys.argv


def fresh_collection():
    return chromadb.EphemeralClient().create_collection(
        f"try_jigo_{int(time.time())}", metadata={"hnsw:space": "cosine"})


col = M.collection if REAL else fresh_collection()
print(f"\nJigo test console | database: {'REAL chroma_memory' if REAL else 'throwaway (in-memory)'} "
      f"| gate {ANSWER_GATE} | /list /reset /quit\n")


def store(text):
    label, inten = text_emotion(text)
    t0 = time.perf_counter()
    mid = M.add_memory(text, source="try_jigo", col=col)
    row = col.get(ids=[mid])["metadatas"][0]
    print(f"  stored  type={row.get('type')}  salience={row.get('salience')}  "
          f"emotion={row.get('emotion', 'none')}"
          f"{'(' + str(row.get('emotion_intensity')) + ')' if row.get('emotion') else ''}  "
          f"[{(time.perf_counter() - t0) * 1000:.0f} ms]")


def recall(q):
    feeling = query_feeling(q)
    res = M.search_memory(q, top_k=8, associative=False, col=col, feeling=feeling)
    if not res:
        print("  (nothing stored yet)  ->", REFUSAL)
        return
    raw = res[0]["raw_similarity"]
    passed = raw >= ANSWER_GATE or res[0].get("emotion_match")
    print(f"  feeling-question={feeling}  top raw similarity={raw:.2f}  gate {ANSWER_GATE}: "
          f"{'PASS' if passed else 'REFUSE (below gate)'}")
    for r in res[:4]:
        tag = f" [{r['emotion']}]" if r.get("emotion") else ""
        print(f"    {r['raw_similarity']:.2f}  score {r['similarity']:.2f}{tag}  {r['content'][:70]}")
    if not passed:
        print("  Jigo:", REFUSAL)
        return
    print("  Jigo:", compose_reply(q, res) or REFUSAL)


while True:
    try:
        line = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue
    if line == "/quit":
        break
    if line == "/reset":
        if REAL:
            print("  /reset is disabled on the real database; delete rows in the dashboard ledger.")
        else:
            col = fresh_collection()
            print("  cleared.")
        continue
    if line == "/list":
        data = col.get()
        for d, m in zip(data["documents"], data["metadatas"]):
            print(f"  - [{m.get('emotion', '-')}] {d[:80]}")
        print(f"  ({col.count()} memories)")
        continue
    if line.lower().startswith("add:"):
        store(line[4:].strip())
    elif looks_like_question(line) or classify_intent_text(line) == "recall":
        recall(line)
    else:
        store(line)
