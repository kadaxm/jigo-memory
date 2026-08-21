"""
Bilingual cross-language retrieval stress test.

Stores 5 memories in English and 5 in Hindi (Roman script, matching how Gemini
transcribes Hindi speech), then runs fixed queries ACROSS languages:
English query -> expect a Hindi-stored memory, and vice versa.

Runs entirely inside a scratch collection (real memory DB is untouched) and
prints a pass/fail table — this is application evidence, not just an internal check.
"""

import sys
import chromadb

from memory import model, add_memory, search_memory

SCRATCH_NAME = "bilingual_test"

ENGLISH_MEMORIES = [
    "My sister is a doctor in Pune",
    "I drink two cups of coffee every morning",
    "The project deadline is next Friday",
    "My laptop password expired last Monday",
    "I prefer window seats on long flights",
]

HINDI_MEMORIES = [
    "Mera naam Rahul hai aur main Delhi me rehta hoon",
    "Mujhe cricket dekhna bahut pasand hai",
    "Main roz subah paanch baje uthta hoon",
    "Mera favorite khana biryani hai",
    "Meri car ka color white hai",
]

# (query, expected_substring, language_direction)
CROSS_QUERIES = [
    ("What is my name and where do I live?", "mera naam rahul", "EN -> HI"),
    ("What food do I like the most?", "biryani", "EN -> HI"),
    ("At what time do I rise every day?", "paanch baje", "EN -> HI"),
    ("What work does my sister do?", "sister is a doctor", "HI -> EN"),
    ("Kitni coffee peeta hoon main roz?", "two cups of coffee", "HI -> EN"),
    ("Project kab tak complete hona chahiye?", "deadline is next friday", "HI -> EN"),
]


def run():
    client = chromadb.PersistentClient(path="./chroma_memory")
    if SCRATCH_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(SCRATCH_NAME)
    scratch = client.get_or_create_collection(name=SCRATCH_NAME, metadata={"hnsw:space": "cosine"})

    print("Storing 5 English memories...")
    for m in ENGLISH_MEMORIES:
        add_memory(m, source="bilingual_test_en", col=scratch)

    print("Storing 5 Hindi (Roman script) memories...")
    for m in HINDI_MEMORIES:
        add_memory(m, source="bilingual_test_hi", col=scratch)

    print("\n" + "=" * 88)
    print(f"{'QUERY':44} {'DIR':8} {'TOP-1':6} {'TOP-3':6} {'RESULT':6}")
    print("=" * 88)

    top1_hits = 0
    top3_hits = 0

    for query, expected, direction in CROSS_QUERIES:
        results = search_memory(query, top_k=3, associative=False, col=scratch)
        contents = [r["content"].lower() for r in results]
        hit1 = bool(contents) and expected in contents[0]
        hit3 = any(expected in c for c in contents)

        top1_hits += hit1
        top3_hits += hit3
        verdict = "PASS" if hit3 else "FAIL"
        t1 = "YES" if hit1 else "no"
        t3 = "YES" if hit3 else "no"
        print(f"{query[:42]:44} {direction:8} {t1:6} {t3:6} {verdict:6}")
        print(f"   expect: '{expected}'")
        if contents:
            print(f"   got   : '{results[0]['content'][:70]}'")

    total = len(CROSS_QUERIES)
    print("=" * 88)
    print(f"TOP-1 accuracy: {top1_hits}/{total} ({100 * top1_hits / total:.0f}%)")
    print(f"TOP-3 accuracy: {top3_hits}/{total} ({100 * top3_hits / total:.0f}%)")
    verdict = "PASS" if top3_hits == total else "PARTIAL/FAIL"
    print(f"CROSS-LANGUAGE RETRIEVAL: {verdict}")

    client.delete_collection(SCRATCH_NAME)
    print("\nScratch collection deleted. Real memory DB untouched.")
    return 0 if top3_hits == total else 1


if __name__ == "__main__":
    sys.exit(run())
