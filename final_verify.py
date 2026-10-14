"""Final end-to-end verification of the ACTUAL demo flow:
confirm-card path (/add) with persona styling — the exact path that failed live."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx  # noqa: E402

BASE = "http://127.0.0.1:8000"

print("=== TEST 1: /add with sad content -> persona reply ===", flush=True)
r = httpx.post(f"{BASE}/add", json={"content": "I am not feeling good today", "source": "voice"}, timeout=60.0)
d = r.json()
print("HTTP", r.status_code, "| id:", (d.get("id") or "")[:8], "| mode:", d.get("reply_mode"), flush=True)
print("reply:", (d.get("reply_text") or "")[:130], flush=True)
t1_pass = d.get("reply_mode") == "persona" and bool(d.get("reply_text"))

print("\n=== TEST 2: /add with happy content -> roast reply ===", flush=True)
r = httpx.post(f"{BASE}/add", json={"content": "Bro, I finally fixed the bug and deployed today", "source": "voice"}, timeout=60.0)
d = r.json()
print("HTTP", r.status_code, "| id:", (d.get("id") or "")[:8], "| mode:", d.get("reply_mode"), flush=True)
print("reply:", (d.get("reply_text") or "")[:130], flush=True)
t2_pass = d.get("reply_mode") == "roast" and bool(d.get("reply_text"))

print("\n=== TEST 3: /add neutral content -> no persona (control) ===", flush=True)
r = httpx.post(f"{BASE}/add", json={"content": "I use ChromaDB for vector storage in my projects", "source": "voice"}, timeout=60.0)
d = r.json()
print("HTTP", r.status_code, "| mode:", d.get("reply_mode"), "| reply_text:", d.get("reply_text"), flush=True)
t3_pass = d.get("reply_mode") is None and d.get("reply_text") is None

# cleanup test memories (keep DB clean)
print("\ncleaning test memories...", flush=True)
import chromadb  # noqa: E402
os.chdir(os.path.dirname(os.path.abspath(__file__)))
c = chromadb.PersistentClient(path="./chroma_memory")
col = [x for x in c.list_collections() if x.name == "memories"][0]
for content in ["I am not feeling good today", "Bro, I finally fixed the bug and deployed today",
                "I use ChromaDB for vector storage in my projects"]:
    rows = col.get(where={"$and": []}) if False else None
# simpler: query by document text
rows = col.get(include=["documents"])
kill = [i for i, doc in zip(rows["ids"], rows["documents"])
        if doc.strip() in ("I am not feeling good today", "Bro, I finally fixed the bug and deployed today",
                           "I use ChromaDB for vector storage in my projects")]
for i in kill:
    col.delete(ids=[i])
print(f"deleted {len(kill)} test memories | final count: {col.count()}", flush=True)

print("\n=== VERDICT ===", flush=True)
print(f"persona on /add: {'PASS' if t1_pass else 'FAIL'}", flush=True)
print(f"roast on /add:   {'PASS' if t2_pass else 'FAIL'}", flush=True)
print(f"neutral control: {'PASS' if t3_pass else 'FAIL'}", flush=True)
print("VERIFY_DONE", flush=True)